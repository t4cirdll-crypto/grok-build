//! Adapter between Grok Build sampler types and the `agentix` LLM client.
//!
//! Implements the ConcentrateAI backend by delegating to agentix's
//! `Provider::SuperCloud` which speaks NDJSON to the Render proxy.
//! This is a self-contained module: no changes needed to the existing
//! stream transforms or Actor infrastructure.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use futures_util::stream::BoxStream;

use xai_grok_sampling_types::{
    AssistantItem, ConversationItem, ConversationRequest, ConversationResponse,
    ContentPart, SamplingError, StopReason, TokenUsage,
    ToolCall as GrokToolCall, ToolSpec,
};

use crate::events::{SamplingChannel, SamplingErrorInfo, SamplingEvent};
use crate::metrics::InferenceLatencyStats;
use crate::types::RequestId;

// Re-export for the request_task dispatch
pub(crate) use xai_grok_sampling_types::ApiBackend;

/// Stream a ConcentrateAI request via agentix, producing [`SamplingEvent`]s.
pub async fn stream_concentrate(
    api_key: String,
    model: String,
    base_url: String,
    request: ConversationRequest,
    request_id: RequestId,
    idle_timeout: Duration,
) -> Result<BoxStream<'static, SamplingEvent>, SamplingError> {
    let stream_start = Instant::now();

    let (messages, system_prompt) = build_agentix_messages(&request.items);
    let tools = build_agentix_tools(&request.tools);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SamplingEvent>();

    let _ = tx.send(SamplingEvent::StreamStarted {
        request_id: request_id.clone(),
        timestamp_ms: chrono::Utc::now().timestamp_millis(),
    });

    let req_id = request_id.clone();
    tokio::spawn(async move {
        drive_agentix(
            api_key, model, base_url, system_prompt, messages, tools,
            tx, req_id, idle_timeout, stream_start,
        ).await;
    });

    Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).boxed())
}

// ── Internal driver ───────────────────────────────────────────────────────────

async fn drive_agentix(
    api_key: String,
    model: String,
    base_url: String,
    system_prompt: Option<String>,
    messages: Vec<agentix::Message>,
    tools: Vec<agentix::raw::shared::ToolDefinition>,
    tx: tokio::sync::mpsc::UnboundedSender<SamplingEvent>,
    request_id: RequestId,
    _idle_timeout: Duration,
    stream_start: Instant,
) {
    let http = match reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(SamplingEvent::Failed {
                request_id,
                error: SamplingErrorInfo::from(&SamplingError::Http(e)),
            });
            return;
        }
    };

    let mut req = agentix::Request::supercloud(api_key)
        .model(model)
        .base_url(base_url);

    if let Some(sys) = system_prompt {
        req = req.system_prompt(sys);
    }
    if !messages.is_empty() {
        req = req.messages(messages);
    }
    if !tools.is_empty() {
        req = req.tools(tools);
    }

    let mut stream = match req.stream(&http).await {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(SamplingEvent::Failed {
                request_id,
                error: SamplingErrorInfo::from(&SamplingError::Api {
                    status: reqwest::StatusCode::BAD_GATEWAY,
                    message: e.to_string(),
                    model_metadata: None,
                    retry_after_secs: None,
                    should_retry: Some(true),
                }),
            });
            return;
        }
    };

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<GrokToolCall> = Vec::new();
    let mut usage: Option<TokenUsage> = None;
    let mut stop_reason: Option<StopReason> = None;
    let mut chunk_index: u64 = 0;
    let mut first_token_seen = false;

    while let Some(event) = stream.next().await {
        match event {
            agentix::LlmEvent::Token(text) => {
                if !first_token_seen {
                    first_token_seen = true;
                    let _ = tx.send(SamplingEvent::FirstToken {
                        request_id: request_id.clone(),
                    });
                }
                content.push_str(&text);
                let _ = tx.send(SamplingEvent::ChannelToken {
                    request_id: request_id.clone(),
                    channel: SamplingChannel::Text,
                    text,
                    chunk_index,
                });
                chunk_index += 1;
            }
            agentix::LlmEvent::Reasoning(text) => {
                reasoning.push_str(&text);
                let _ = tx.send(SamplingEvent::ChannelToken {
                    request_id: request_id.clone(),
                    channel: SamplingChannel::Reasoning,
                    text,
                    chunk_index,
                });
                chunk_index += 1;
            }
            agentix::LlmEvent::ToolCall(tc) => {
                let tool_call = GrokToolCall {
                    id: Arc::from(tc.id.as_str()),
                    name: tc.name,
                    arguments: Arc::from(tc.arguments.as_str()),
                };
                tool_calls.push(tool_call.clone());
                let _ = tx.send(SamplingEvent::ToolCallDelta {
                    request_id: request_id.clone(),
                    tool_index: (tool_calls.len() - 1) as u32,
                    id: Some(tool_call.id.to_string()),
                    name: Some(tool_call.name.clone()),
                    arguments_delta: Some(tool_call.arguments.to_string()),
                });
            }
            agentix::LlmEvent::ToolCallChunk(tcc) => {
                let _ = tx.send(SamplingEvent::ToolCallDelta {
                    request_id: request_id.clone(),
                    tool_index: tcc.index,
                    id: if tcc.id.is_empty() { None } else { Some(tcc.id) },
                    name: if tcc.name.is_empty() { None } else { Some(tcc.name) },
                    arguments_delta: if tcc.delta.is_empty() { None } else { Some(tcc.delta) },
                });
            }
            agentix::LlmEvent::Usage(u) => {
                usage = Some(TokenUsage {
                    prompt_tokens: u.prompt_tokens as u32,
                    completion_tokens: u.completion_tokens as u32,
                    total_tokens: u.total_tokens as u32,
                    ..Default::default()
                });
            }
            agentix::LlmEvent::Done => {
                break;
            }
            agentix::LlmEvent::Error(e) => {
                let _ = tx.send(SamplingEvent::Failed {
                    request_id: request_id.clone(),
                    error: SamplingErrorInfo::from(&SamplingError::Api {
                        status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                        message: e,
                        model_metadata: None,
                        retry_after_secs: None,
                        should_retry: Some(true),
                    }),
                });
                return;
            }
            _ => {}
        }
    }

    // Determine stop reason
    stop_reason = if tool_calls.is_empty() {
        Some(StopReason::Stop)
    } else {
        Some(StopReason::ToolCalls)
    };

    // Build ConversationResponse
    let mut items: Vec<ConversationItem> = Vec::new();

    let assistant_content = if reasoning.is_empty() {
        content.clone()
    } else {
        format!("{}\n\n<reasoning>{}</reasoning>", content, reasoning)
    };

    items.push(ConversationItem::Assistant(AssistantItem {
        content: Arc::from(assistant_content.as_str()),
        tool_calls,
        model_id: None,
        model_fingerprint: None,
        reasoning_effort: None,
    }));

    let elapsed_ms = stream_start.elapsed().as_millis() as u64;

    let response = ConversationResponse {
        items,
        stop_reason,
        usage,
        cost_usd_ticks: None,
        message_chunks_emitted: chunk_index,
        doom_loop_signals: vec![],
        stop_message: None,
    };

    let metrics = InferenceLatencyStats {
        time_to_first_token_ms: None,
        time_to_last_byte_ms: elapsed_ms,
        chunk_count: chunk_index as u32,
        itl_intervals_ms: vec![],
        itl_p50_ms: None,
        itl_p99_ms: None,
        itl_max_ms: None,
        itl_mean_ms: None,
        attempts: 1,
    };

    let _ = tx.send(SamplingEvent::Completed {
        request_id,
        response: Box::new(response),
        metrics,
    });
}

// ── Type conversions ──────────────────────────────────────────────────────────

/// Convert Grok Build ConversationItems to agentix Messages.
fn build_agentix_messages(items: &[ConversationItem]) -> (Vec<agentix::Message>, Option<String>) {
    let mut messages = Vec::new();
    let mut system_prompt: Option<String> = None;

    for item in items {
        match item {
            ConversationItem::System(sys) => {
                system_prompt = Some(sys.content.to_string());
            }
            ConversationItem::User(user) => {
                let mut content = Vec::new();
                for part in &user.content {
                    match part {
                        ContentPart::Text { text } => {
                            content.push(agentix::Content::text(text.to_string()));
                        }
                        ContentPart::Image { url } => {
                            // agentix doesn't have a direct ImageContent builder
                            // publicly accessible, so use text placeholder.
                            let _ = url; // suppress unused warning
                            content.push(agentix::Content::text("[image]"));
                        }
                    }
                }
                if !content.is_empty() {
                    messages.push(agentix::Message::User(content));
                }
            }
            ConversationItem::Assistant(assistant) => {
                let tool_calls = assistant
                    .tool_calls
                    .iter()
                    .map(|tc| agentix::request::ToolCall {
                        id: tc.id.to_string(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    })
                    .collect();

                messages.push(agentix::Message::Assistant {
                    content: if assistant.content.is_empty() {
                        None
                    } else {
                        Some(assistant.content.to_string())
                    },
                    reasoning: None,
                    tool_calls,
                    provider_data: None,
                });
            }
            ConversationItem::ToolResult(tool_result) => {
                messages.push(agentix::Message::ToolResult {
                    call_id: tool_result.tool_call_id.clone(),
                    content: vec![agentix::Content::text(tool_result.content.to_string())],
                });
            }
            _ => {
                // BackendToolCall, Reasoning — skip for now
            }
        }
    }

    (messages, system_prompt)
}

/// Convert Grok Build ToolSpecs to agentix ToolDefinitions.
fn build_agentix_tools(tools: &[ToolSpec]) -> Vec<agentix::raw::shared::ToolDefinition> {
    tools
        .iter()
        .map(|t| agentix::raw::shared::ToolDefinition {
            kind: agentix::raw::shared::ToolKind::Function,
            function: agentix::raw::shared::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
                strict: None,
            },
        })
        .collect()
}
