
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;
using System.Linq;

namespace Aurora.LLM
{
    /// <summary>
    /// Cliente genérico para LLMs con contrato OpenAI-compatible y Azure OpenAI.
    /// Diseñado para .NET Framework (4.6.2+). Requiere Newtonsoft.Json.
    /// </summary>
    public sealed class LLMGeneric
    {
        // -----------------------------
        // Tipos públicos (DTOs/Contratos)
        // -----------------------------

        public enum LLMProvider
        {
            OpenAI,            // OpenAI u otro servidor OpenAI-compatible
            OpenAICompatible,  // Alias del anterior
            AzureOpenAI        // Azure OpenAI (deployment como "model")
        }

        public enum LLMRole { System, User, Assistant, Tool }

        public sealed class LLMMessage
        {
            [JsonProperty("role")]
            public string Role { get; set; }  // "system","user","assistant","tool"

            [JsonProperty("content")]
            public string Content { get; set; }

            public LLMMessage() { }

            public LLMMessage(LLMRole role, string content)
            {
                switch (role)
                {
                    case LLMRole.System: Role = "system"; break;
                    case LLMRole.User: Role = "user"; break;
                    case LLMRole.Assistant: Role = "assistant"; break;
                    case LLMRole.Tool: Role = "tool"; break;
                    default: Role = "user"; break;
                }
                Content = content ?? string.Empty;
            }
        }

        public sealed class LLMRequest
        {
            /// <summary>Si no especificas Messages, se usará Prompt como único mensaje de usuario.</summary>
            public string Prompt { get; set; }
            public List<LLMMessage> Messages { get; set; }
            public double? Temperature { get; set; }
            public int? MaxTokens { get; set; }

            /// <summary>Parámetros adicionales OpenAI-style: top_p, presence_penalty, frequency_penalty, tools, response_format, etc.</summary>
            public Dictionary<string, object> Extra { get; set; }

            public LLMRequest()
            {
                Messages = new List<LLMMessage>();
                Extra = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
            }
        }

        public sealed class LLMError
        {
            public string Code { get; set; }
            public string Message { get; set; }
            public int? StatusCode { get; set; }
            public string RawBody { get; set; }

            public override string ToString()
            {
                return string.Format("{0} (code={1}, status={2})", Message, Code, StatusCode.HasValue ? StatusCode.Value.ToString() : "null");
            }
        }

        public sealed class LLMResponse
        {
            public string Text { get; set; }
            public string FinishReason { get; set; }
            public LLMError Error { get; set; }
            public string RawJson { get; set; }
            public bool IsSuccess { get { return Error == null; } }
        }

        public sealed class LLMClientOptions
        {
            /// <summary>Proveedor: OpenAI/OpenAICompatible (incluye LM Studio, Ollama con proxy OpenAI), o AzureOpenAI.</summary>
            public LLMProvider Provider { get; set; }

            /// <summary>Clave de API. En locales (LM Studio/Ollama proxy) puede ser null.</summary>
            public string ApiKey { get; set; }

            /// <summary>Modelo o DeploymentId (en Azure). Ej: "gpt-4o-mini" o "gpt4o-mini-deploy".</summary>
            public string Model { get; set; }

            /// <summary>Base URL. OpenAI: "https://api.openai.com". LM Studio: "http://localhost:1234". Ollama proxy: "http://localhost:11434".</summary>
            public string BaseUrl { get; set; }

            /// <summary>Solo Azure OpenAI: versión del API.</summary>
            public string ApiVersion { get; set; }

            /// <summary>Tiempo máximo por request.</summary>
            public TimeSpan Timeout { get; set; }

            /// <summary>Reintentos ante fallos transitorios (>=500 o timeout).</summary>
            public int MaxRetries { get; set; }

            /// <summary>Cuando la respuesta viene vacía, retornar error estándar.</summary>
            public bool FailOnEmptyContent { get; set; }

            /// <summary>Headers adicionales.</summary>
            public Dictionary<string, string> AdditionalHeaders { get; set; }

            public LLMClientOptions()
            {
                Provider = LLMProvider.OpenAI;
                BaseUrl = "https://api.openai.com";
                ApiVersion = "2024-02-15-preview"; // Azure
                Timeout = TimeSpan.FromSeconds(45);
                MaxRetries = 2;
                FailOnEmptyContent = true;
                AdditionalHeaders = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            }
        }

        // -----------------------------
        // Campos/Estado
        // -----------------------------

        private static readonly HttpClient SharedHttpClient = new HttpClient();
        private readonly LLMClientOptions _opt;

        // -----------------------------
        // Ctor
        // -----------------------------

        public LLMGeneric(LLMClientOptions options)
        {
            if (options == null) throw new ArgumentNullException("options");
            if (string.IsNullOrEmpty(options.Model)) throw new ArgumentException("Model no puede ser null/empty.", "options.Model");
            if (string.IsNullOrEmpty(options.BaseUrl)) throw new ArgumentException("BaseUrl no puede ser null/empty.", "options.BaseUrl");

            _opt = options;
            SharedHttpClient.Timeout = _opt.Timeout;
        }

        // -----------------------------
        // API principal
        // -----------------------------

        /// <summary>
        /// Genera una respuesta de chat usando el proveedor configurado.
        /// Compatible con /v1/chat/completions (OpenAI) y Azure OpenAI (deployments).
        /// </summary>
        public async Task<LLMResponse> GenerateChatAsync(LLMRequest request, CancellationToken ct)
        {
            if (request == null) throw new ArgumentNullException("request");

            // Normaliza mensajes a partir de Prompt si hace falta
            List<LLMMessage> messages = (request.Messages != null && request.Messages.Count > 0)
                ? request.Messages
                : new List<LLMMessage> { new LLMMessage(LLMRole.User, request.Prompt ?? string.Empty) };

            string url = BuildChatCompletionsUrl();

            // Payload OpenAI-style
            var payload = new Dictionary<string, object>();
            payload["model"] = _opt.Provider == LLMProvider.AzureOpenAI ? _opt.Model : _opt.Model;
            payload["messages"] = messages.Select(m =>
            {
                var d = new Dictionary<string, object>();
                d["role"] = string.IsNullOrEmpty(m.Role) ? "user" : m.Role;
                d["content"] = m.Content ?? string.Empty;
                return d;
            }).ToList();

            if (request.Temperature.HasValue) payload["temperature"] = request.Temperature.Value;
            if (request.MaxTokens.HasValue) payload["max_tokens"] = request.MaxTokens.Value;

            if (request.Extra != null)
            {
                foreach (var kv in request.Extra)
                {
                    payload[kv.Key] = kv.Value;
                }
            }

            string json = JsonConvert.SerializeObject(payload);
            var httpReq = new HttpRequestMessage(HttpMethod.Post, url);
            httpReq.Content = new StringContent(json, Encoding.UTF8, "application/json");

            // Headers
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

            if (_opt.AdditionalHeaders != null)
            {
                foreach (var kv in _opt.AdditionalHeaders)
                    httpReq.Headers.TryAddWithoutValidation(kv.Key, kv.Value);
            }

            int attempt = 0;
            for (; ; )
            {
                attempt++;
                HttpResponseMessage resp = null;
                string body = null;

                try
                {
                    resp = await SharedHttpClient.SendAsync(Clone(httpReq), ct).ConfigureAwait(false);
                    body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);

                    if (!resp.IsSuccessStatusCode)
                        return ParseError(body, (int)resp.StatusCode);

                    // Parse éxito: OpenAI style => choices[0].message.content
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

                    LLMResponse ok = new LLMResponse();
                    ok.Text = (text ?? string.Empty).Trim();
                    ok.FinishReason = finish;
                    ok.RawJson = body;
                    return ok;
                }
                catch (TaskCanceledException ex)
                {
                    // Timeout (si no es cancelación explícita del token)
                    if (ct.IsCancellationRequested || attempt > _opt.MaxRetries)
                    {
                        return new LLMResponse
                        {
                            Error = new LLMError
                            {
                                Code = "timeout",
                                Message = "Tiempo de espera agotado al consultar el modelo.",
                                StatusCode = null,
                                RawBody = ex.ToString()
                            }
                        };
                    }
                    await Task.Delay(Backoff(attempt), ct).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    if (attempt > _opt.MaxRetries)
                    {
                        return new LLMResponse
                        {
                            Error = new LLMError
                            {
                                Code = "unhandled_exception",
                                Message = "Error inesperado al consultar el modelo.",
                                StatusCode = null,
                                RawBody = ex.ToString()
                            }
                        };
                    }
                    await Task.Delay(Backoff(attempt), ct).ConfigureAwait(false);
                }
            }
        }

        public Task<LLMResponse> GenerateChatAsync(LLMRequest request)
        {
            return GenerateChatAsync(request, CancellationToken.None);
        }

        // -----------------------------
        // Helpers privados
        // -----------------------------

        private string BuildChatCompletionsUrl()
        {
            string baseUrl = (_opt.BaseUrl ?? string.Empty).Trim().TrimEnd('/');

            if (_opt.Provider == LLMProvider.AzureOpenAI)
            {
                // Azure: {BaseUrl}/openai/deployments/{deploymentId}/chat/completions?api-version=...
                return string.Format("{0}/openai/deployments/{1}/chat/completions?api-version={2}",
                    baseUrl, _opt.Model, string.IsNullOrEmpty(_opt.ApiVersion) ? "2024-02-15-preview" : _opt.ApiVersion);
            }

            // OpenAI-compatible: {BaseUrl}/v1/chat/completions (evita duplicar /v1)
            if (baseUrl.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
                return baseUrl + "/chat/completions";

            return baseUrl + "/v1/chat/completions";
        }

        private static TimeSpan Backoff(int attempt)
        {
            // backoff exponencial simple
            int pow = attempt;
            if (pow > 4) pow = 4;
            double ms = 250 * Math.Pow(2, pow);
            return TimeSpan.FromMilliseconds(ms);
        }

        private static LLMResponse ParseError(string body, int statusCode)
        {
            try
            {
                dynamic dyn = JsonConvert.DeserializeObject(body);
                string message = null;
                string code = null;

                if (dyn != null && dyn.error != null)
                {
                    message = (string)dyn.error.message;
                    code = (string)dyn.error.code;
                }

                if (string.IsNullOrWhiteSpace(message))
                {
                    // Fallback: recorta body
                    if (!string.IsNullOrEmpty(body))
                        message = body.Length > 400 ? (body.Substring(0, 400) + "...") : body;
                    else
                        message = "Error al consultar el modelo.";
                }
                if (string.IsNullOrWhiteSpace(code)) code = "api_error";

                LLMResponse r = new LLMResponse();
                r.Error = new LLMError
                {
                    Code = code,
                    Message = message,
                    StatusCode = statusCode,
                    RawBody = body
                };
                r.RawJson = body;
                return r;
            }
            catch
            {
                LLMResponse r = new LLMResponse();
                r.Error = new LLMError
                {
                    Code = "api_error",
                    Message = "Error al consultar el modelo (no se pudo interpretar la respuesta).",
                    StatusCode = statusCode,
                    RawBody = body
                };
                r.RawJson = body;
                return r;
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


