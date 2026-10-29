def kv_cache_gb(layers, kv_heads, head_dim, seq, batch=1, bytes_=2):
    return 2 * layers * kv_heads * head_dim * seq * batch * bytes_ / 1e9
def weights_gb(params_b, bits): return params_b * 1e9 * bits / 8 / 1e9
print("Llama-3-8B-like (32 layers, 8 KV heads (GQA), head_dim 128):")
for seq in [8_192, 32_768, 128_000]: print(f"  KV-cache @ {seq:>7,} tokens, batch 1, fp16: {kv_cache_gb(32, 8, 128, seq):6.2f} GB")
for bits in [16, 8, 4]: print(f"  weights at {bits:2d}-bit: {weights_gb(8, bits):5.1f} GB")
d = k = 4096; r = 16
print(f"LoRA r={r} on a {d}x{k} matrix: {r*(d+k):,} trainable vs {d*k:,} ({r*(d+k)/(d*k):.2%})")
tokens_in, tokens_out, p_in, p_out = 3_000, 400, 0.15, 0.60        # USD per 1M tokens (example prices)
print(f"cost per request ≈ USD {(tokens_in*p_in + tokens_out*p_out)/1e6:.5f}; per 1M requests ≈ USD {(tokens_in*p_in + tokens_out*p_out):,.0f}")
