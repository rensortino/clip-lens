import torch
from transformers import AutoImageProcessor, AutoModel, BlipForConditionalGeneration, BlipProcessor, AutoTokenizer

class EmbeddingModel:
    def __init__(self, model_id, device="cuda"):
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.preprocessor = AutoImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.model.eval()

    def __call__(self, x):
        return self.encode_image(x)
    
    def encode_image(self, x):
        with torch.no_grad():
            sample = self.preprocessor(x, return_tensors="pt").to(self.device)
            if "dino" in str(self.model.__class__).lower():
                # In DINOv2, pooler_output == last_hidden_state[:,0], while in ViT the weights of ViTPooler are not initialized, 
                # so we need to explicitly take the class token
                return self.model(**sample).pooler_output 
            elif "clip" in str(self.model.__class__).lower():
                return self.model.get_image_features(**sample)
            elif "vit" in str(self.model.__class__).lower():
                return self.model(**sample).last_hidden_state[:, 0]
            else:
                raise ValueError(f"Model {self.model} does not support image encoding")
            
    def encode_text(self, x):
        with torch.no_grad():
            sample = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
            if "clip" in str(self.model.__class__).lower():
                return self.model.get_text_features(**sample)
            else:
                raise ValueError(f"Model {self.model} does not support text encoding")
            





class CaptioningModel:
    def __init__(self, model_id="microsoft/Florence-2-base", device="cuda"):
        self.model = BlipForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.preprocessor = BlipProcessor.from_pretrained(model_id)
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # self.emb_size = self.encode_image(torch.rand([1,3,224,224], device=self.device)).shape[-1]
        self.model.eval()

    def __call__(self, x):
        return self.get_caption(x)
    
    def blip_caption(self, x):
        inputs = self.preprocessor(x, return_tensors="pt").to("cuda", torch.float16)

        out = self.model.generate(**inputs)
        return self.preprocessor.decode(out[0], skip_special_tokens=True)
    
    def florence_caption(self, image, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.preprocessor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
        )
        generated_text = self.preprocessor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.preprocessor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

        return parsed_answer
    
    def get_caption(self, x):
        with torch.no_grad():
            if "Florence" in str(self.model.__class__).lower():
                prompt = "<CAPTION>"
                return self.florence_caption(x, prompt)
            if "blip" in str(self.model.__class__).lower():
                return self.blip_caption(x)
            else:
                raise ValueError(f"Model {self.model} not supported")