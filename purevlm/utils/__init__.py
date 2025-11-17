try:
    import flash_attn
    flash_attn_available = True
except ImportError:
    flash_attn_available = False