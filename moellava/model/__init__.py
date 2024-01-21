from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig
from .language_model.llava_llama_moe import MoELLaVALlamaForCausalLM, MoELLaVALlamaConfig
from .language_model.llava_qwen import LlavaQWenForCausalLM, LlavaQWenConfig
from .language_model.llava_qwen_moe import MoELLaVAQWenForCausalLM, MoELLaVAQWenConfig
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
    from .language_model.llava_mistral_moe import MoELLaVAMistralForCausalLM, MoELLaVAMistralConfig
if a == '4' and int(b) >= 36:
    from .language_model.llava_phi import LlavaPhiForCausalLM, LlavaPhiConfig
    from .language_model.llava_phi_moe import MoELLaVAPhiForCausalLM, MoELLaVAPhiConfig
    from .language_model.llava_stablelm import LlavaStablelmForCausalLM, LlavaStablelmConfig
    from .language_model.llava_stablelm_moe import MoELLaVAStablelmForCausalLM, MoELLaVAStablelmConfig
if a == '4' and int(b) <= 31:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
