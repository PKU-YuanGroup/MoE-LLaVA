from .model import LlavaLlamaForCausalLM
from .model import MoELLaVALlamaForCausalLM
from .model import LlavaQWenForCausalLM
from .model import MoELLaVALlamaForCausalLM
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .model import LlavaMistralForCausalLM
    from .model import MoELLaVAMistralForCausalLM
if a == '4' and int(b) >= 36:
    from .model import LlavaMiniCPMForCausalLM
    from .model import MoELLaVAMiniCPMForCausalLM
    from .model import LlavaPhiForCausalLM
    from .model import MoELLaVAPhiForCausalLM
    from .model import LlavaStablelmForCausalLM
    from .model import MoELLaVAStablelmForCausalLM
if a == '4' and int(b) >= 37:
    from .model import LlavaQwen1_5ForCausalLM
    from .model import MoELLaVAQwen1_5ForCausalLM
