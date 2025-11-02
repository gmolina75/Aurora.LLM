using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;
using System.Linq;

namespace Aurora.LLM
{
    public sealed class LLMGeneric
    {
        public enum LLMProvider { OpenAI, OpenAICompatible, AzureOpenAI }
        public enum LLMRole { System, User, Assistant, Tool }

        public sealed class LLMMessage
        {
            [JsonProperty("role")] public string Role { get; set; }
            [JsonProperty("content")] public string Content { get; set; }
            public LLMMessage() { }
            public LLMMessage(LLMRole role, string content)
            {
                Role = role == LLMRole.System ? "system" :
                       role == LLMRole.Assistant ? "assistant" :
                       role == LLMRole.Tool ? "tool" : "user";
                Content = content ?? string.Empty;
            }
        }

        public sealed class LLMRequest
        {
            public string Prompt { get; set; }
            public List<LLMMessage> Messages { get; set; } = new List<LLMMessage>();
            public double? Temperature { get; set; }
            public int? MaxTokens { get; set; }
            public Dictionary<string, object> Extra { get; set; } = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        }

        public sealed class LLMError
        {
            public string Code { get; set; }
            public string Message { get; set; }
            public int? StatusCode { get; set; }
            public string RawBody { get; set; }
            public override string ToString() => $"{Message} (code={Code}, status={(StatusCode.HasValue ? StatusCode.Value.ToString() : "null")})";
        }

        public sealed class LLMResponse
        {
            public string Text { get; set; }
            public string FinishReason { get; set; }
            public LLMError Error { get; set; }
            public string RawJson { get; set; }
            public bool IsSuccess => Error == null;
        }

        public sealed class LLMClientOptions
        {
            public LLMProvider Provider { get; set; } = LLMProvider.OpenAI;
            public string ApiKey { get; set; }
            public string Model { get; set; }
            public string BaseUrl { get; set; } = "https://api.openai.com";
            public string ApiVersion { get; set; } = "2024-02-15-preview"; // Azure
            public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(45);
            public int MaxRetries { get; set; } = 2;
            public bool FailOnEmptyContent { get; set; } = true;
            public Dictionary<string, string> AdditionalHeaders { get; set; } = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        private static readonly HttpClient SharedHttpClient = new HttpClient(); // NO se toca Timeout
        private readonly LLMClientOptions _opt;

        public LLMGeneric(LLMClientOptions options)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));
            if (string.IsNullOrEmpty(options.Model)) throw new ArgumentException("Model no puede ser null/empty.", nameof(options.Model));
            if (string.IsNullOrEmpty(options.BaseUrl)) throw new ArgumentException("BaseUrl no puede ser null/empty.", nameof(options.BaseUrl));
            _opt = options;
        }

        public Task<LLMResponse> GenerateChatAsync(LLMRequest request)
            => GenerateChatAsync(request, CancellationToken.None);

        public async Task<LLMResponse> GenerateChatAsync(LLMRequest request, CancellationToken ct)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));

            var messages = (request.Messages != null && request.Messages.Count > 0)
                ? request.Messages
                : new List<LLMMessage> { new LLMMessage(LLMRole.User, request.Prompt ?? string.Empty) };

            var url = BuildChatCompletionsUrl();

            var payload = new Dictionary<string, object>
            {
                ["model"] = _opt.Model,
                ["messages"] = messages.Select(m => new Dictionary<string, object>
                {
                    ["role"] = string.IsNullOrEmpty(m.Role) ? "user" : m.Role,
                    ["content"] = m.Content ?? string.Empty
                }).ToList()
            };
            if (request.Temperature.HasValue) payload["temperature"] = request.Temperature.Value;
            if (request.MaxTokens.HasValue) payload["max_tokens"] = request.MaxTokens.Value;
            foreach (var kv in request.Extra) payload[kv.Key] = kv.Value;

            var httpReq = new HttpRequestMessage(HttpMethod.Post, url)
            {
                Content = new StringContent(JsonConvert.SerializeObject(payload), Encoding.UTF8, "application/json")
            };

            if (_opt.Provider == LLMProvider.AzureOpenAI)
            {
                if (!string.IsNullOrWhiteSpace(_opt.ApiKey))
                    httpReq.Headers.TryAddWithoutValidation("api-key", _opt.ApiKey);
            }
            else
            {
                if (!string.IsNullOrWhiteSpace(_opt.ApiKey))
                    httpReq.Headers.TryAddWithoutValidation("Authorization", "Bearer " + _opt.ApiKey);
            }
            foreach (var kv in _opt.AdditionalHeaders)
                httpReq.Headers.TryAddWithoutValidation(kv.Key, kv.Value);

            int attempt = 0;
            for (; ; )
            {
                attempt++;
                HttpResponseMessage resp = null;
                string body = null;

                // --- Timeout por solicitud ---
                using (var linkCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    linkCts.CancelAfter(_opt.Timeout);
                    try
                    {
                        resp = await SharedHttpClient.SendAsync(Clone(httpReq), linkCts.Token).ConfigureAwait(false);
                        body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);

                        if (!resp.IsSuccessStatusCode)
                            return ParseError(body, (int)resp.StatusCode);

                        dynamic parsed = JsonConvert.DeserializeObject(body);
                        string text = (string)(parsed != null && parsed.choices != null && parsed.choices.Count > 0
                            ? parsed.choices[0].message.content
                            : "");
                        string finish = (string)(parsed != null && parsed.choices != null && parsed.choices.Count > 0
                            ? parsed.choices[0].finish_reason
                            : "");

                        if (string.IsNullOrWhiteSpace(text) && _opt.FailOnEmptyContent)
                        {
                            return new LLMResponse
                            {
                                Error = new LLMError
                                {
                                    Code = "empty_content",
                                    Message = "El modelo respondió vacío. Posible saturación o no disponible.",
                                    StatusCode = (int)resp.StatusCode,
                                    RawBody = body
                                },
                                RawJson = body
                            };
                        }

                        return new LLMResponse
                        {
                            Text = (text ?? string.Empty).Trim(),
                            FinishReason = finish,
                            RawJson = body
                        };
                    }
                    catch (TaskCanceledException ex)
                    {
                        if (linkCts.IsCancellationRequested && !ct.IsCancellationRequested)
                        {
                            // Timeout propio (no cancelación externa)
                            if (attempt > _opt.MaxRetries)
                                return new LLMResponse { Error = new LLMError { Code = "timeout", Message = "Tiempo de espera agotado al consultar el modelo.", RawBody = ex.ToString() } };
                            await Task.Delay(Backoff(attempt), ct).ConfigureAwait(false);
                            continue;
                        }

                        // Cancelación externa o agotó reintentos
                        if (attempt > _opt.MaxRetries)
                            return new LLMResponse { Error = new LLMError { Code = "timeout", Message = "Tiempo de espera agotado al consultar el modelo.", RawBody = ex.ToString() } };
                        await Task.Delay(Backoff(attempt), ct).ConfigureAwait(false);
                    }
                    catch (Exception ex)
                    {
                        if (attempt > _opt.MaxRetries)
                            return new LLMResponse { Error = new LLMError { Code = "unhandled_exception", Message = "Error inesperado al consultar el modelo.", RawBody = ex.ToString() } };
                        await Task.Delay(Backoff(attempt), ct).ConfigureAwait(false);
                    }
                }
            }
        }

        private string BuildChatCompletionsUrl()
        {
            string baseUrl = (_opt.BaseUrl ?? string.Empty).Trim().TrimEnd('/');
            if (_opt.Provider == LLMProvider.AzureOpenAI)
                return $"{baseUrl}/openai/deployments/{_opt.Model}/chat/completions?api-version={(_opt.ApiVersion ?? "2024-02-15-preview")}";
            if (baseUrl.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
                return baseUrl + "/chat/completions";
            return baseUrl + "/v1/chat/completions";
        }

        private static TimeSpan Backoff(int attempt)
        {
            int pow = Math.Min(attempt, 4);
            return TimeSpan.FromMilliseconds(250 * Math.Pow(2, pow));
        }

        private static LLMResponse ParseError(string body, int statusCode)
        {
            try
            {
                dynamic dyn = JsonConvert.DeserializeObject(body);
                string message = null, code = null;
                if (dyn != null && dyn.error != null)
                {
                    message = (string)dyn.error.message;
                    code = (string)dyn.error.code;
                }
                if (string.IsNullOrWhiteSpace(message))
                    message = !string.IsNullOrEmpty(body) ? (body.Length > 400 ? body.Substring(0, 400) + "..." : body) : "Error al consultar el modelo.";
                if (string.IsNullOrWhiteSpace(code)) code = "api_error";

                return new LLMResponse
                {
                    Error = new LLMError { Code = code, Message = message, StatusCode = statusCode, RawBody = body },
                    RawJson = body
                };
            }
            catch
            {
                return new LLMResponse
                {
                    Error = new LLMError { Code = "api_error", Message = "Error al consultar el modelo (no se pudo interpretar la respuesta).", StatusCode = statusCode, RawBody = body },
                    RawJson = body
                };
            }
        }

        private static HttpRequestMessage Clone(HttpRequestMessage req)
        {
            var clone = new HttpRequestMessage(req.Method, req.RequestUri);
            foreach (var header in req.Headers)
                clone.Headers.TryAddWithoutValidation(header.Key, header.Value);

            if (req.Content != null)
            {
                var ms = new MemoryStream();
                req.Content.CopyToAsync(ms).GetAwaiter().GetResult();
                ms.Position = 0;
                var contentClone = new StreamContent(ms);
                foreach (var header in req.Content.Headers)
                    contentClone.Headers.TryAddWithoutValidation(header.Key, header.Value);
                clone.Content = contentClone;
            }
            return clone;
        }
    }
}
