from lvlm.Qwen2FVL import Qwen2FVL


class HuatuoGPTVision(Qwen2FVL):

    def _get_temp(self, temp):
        return max(temp, 0.01)

    def _model_id(self):
        if "/" in self.version:
            return self.version
        # Prefer the HF-native Qwen2.5-VL checkpoint over the older LLaVA-based release.
        if self.version == "HuatuoGPT-Vision-7B":
            return "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
        return f"FreedomIntelligence/{self.version}"
