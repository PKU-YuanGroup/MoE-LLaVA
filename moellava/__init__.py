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
    from .model import LlavaPhiForCausalLM
    from .model import MoELLaVAPhiForCausalLM
    from .model import LlavaStablelmForCausalLM
    from .model import MoELLaVAStablelmForCausalLM