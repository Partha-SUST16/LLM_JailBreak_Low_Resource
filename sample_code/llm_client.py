import re
import time
import torch
import xml.etree.ElementTree as ET
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class BaseLLMClient(ABC):
    
    def __init__(self, model_name: str, enable_thinking: bool = False, batch_size: int = 2,
                 max_new_tokens: int = 5500, temperature: float = 0.6, top_p: float = 0.95,
                 top_k: int = 20, min_p: float = 0.0, max_retries: int = 2):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_retries = max_retries
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        pass
    
    @abstractmethod
    def generate_response(self, user_input: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def generate_batch_response(self, user_inputs: List[str]) -> List[Optional[str]]:
        pass
    
    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                self._clear_memory()
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    return None


class QwenChatbot(BaseLLMClient):
    
    def _initialize_model(self):
        logger.info(f"Loading Qwen model from {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Qwen model loaded successfully")
    
    def generate_response(self, user_input: str) -> Optional[str]:
        def _generate():
            messages = [{"role": "user", "content": user_input}]
            logger.debug(f"Generating response for input: {user_input[:100]}...")
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            start_time = time.time()
            with torch.no_grad():
                response_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p
                )[0][len(inputs.input_ids[0]):].tolist()
            
            end_time = time.time()
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            try:
                index = len(response_ids) - response_ids[::-1].index(151668)
            except ValueError:
                if self.enable_thinking:
                    logger.debug("No </think> found in the output")
                index = 0

            thinking_content = self.tokenizer.decode(response_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(response_ids[index:], skip_special_tokens=True).strip("\n")
            logger.debug(f"LLM thinking content: {thinking_content[:100]}...")
            logger.debug(f"LLM content: {content[:100]}...")
            logger.debug(f"LLM response generated in {end_time - start_time:.2f}s")
            return content.strip()
        
        return self._retry_with_backoff(_generate)
    
    def generate_batch_response(self, user_inputs: List[str]) -> List[Optional[str]]:
        def _generate_batch():
            logger.info(f"Generating batch responses for {len(user_inputs)} inputs")
            
            all_texts = []
            for user_input in user_inputs:
                messages = [{"role": "user", "content": user_input}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking
                )
                all_texts.append(text)
            
            inputs = self.tokenizer(all_texts, return_tensors="pt", padding=True).to(self.model.device)
            
            start_time = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p
                )
            
            end_time = time.time()
            
            responses = []
            for i, generated_id in enumerate(generated_ids):
                input_length = inputs.input_ids[i].shape[0]
                response_ids = generated_id[input_length:].tolist()
                
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                try:
                    index = len(response_ids) - response_ids[::-1].index(151668)
                except ValueError:
                    if self.enable_thinking:
                        logger.debug(f"No </think> found in output {i}")
                    index = 0
                
                thinking_content = self.tokenizer.decode(response_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(response_ids[index:], skip_special_tokens=True).strip("\n")
                
                responses.append(content.strip())
            
            logger.info(f"Batch response generated in {end_time - start_time:.2f}s for {len(user_inputs)} inputs")
            return responses
        
        return self._retry_with_backoff(_generate_batch) or [None] * len(user_inputs)


class QwenChatbotV2(BaseLLMClient):
    
    def _initialize_model(self):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm is not installed. Please install it to use QwenChatbotV2.")
        
        logger.info(f"Loading Qwen model from {self.model_name} using vLLM")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.llm = LLM(model=self.model_name, max_num_seqs=self.batch_size)
        self.SamplingParams = SamplingParams
        
        logger.info("Qwen model loaded successfully with vLLM")
    
    def generate_response(self, user_input: str) -> Optional[str]:
        from vllm import SamplingParams
        
        def _generate():
            messages = [{"role": "user", "content": user_input}]
            logger.debug(f"Generating response for input: {user_input[:100]}...")
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            
            params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
            )
            
            start_time = time.time()
            outputs = self.llm.generate([prompt], sampling_params=params)
            end_time = time.time()
            
            output_text = outputs[0].outputs[0].text.strip()
            
            response_ids = self.tokenizer(output_text, return_tensors="pt").input_ids[0].tolist()
            
            try:
                end_thinking_token_id = 151668
                index = len(response_ids) - response_ids[::-1].index(end_thinking_token_id)
            except ValueError:
                if self.enable_thinking:
                    logger.debug("No </think> found in the output")
                index = 0
            
            thinking_content = self.tokenizer.decode(response_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(response_ids[index:], skip_special_tokens=True).strip("\n")
            
            logger.debug(f"LLM thinking content: {thinking_content[:100]}...")
            logger.debug(f"LLM content: {content[:100]}...")
            logger.debug(f"LLM response generated in {end_time - start_time:.2f}s")
            
            return content.strip()
        
        return self._retry_with_backoff(_generate)
    
    def generate_batch_response(self, user_inputs: List[str]) -> List[Optional[str]]:
        from vllm import SamplingParams
        
        def _generate_batch():
            logger.info(f"Generating batch responses for {len(user_inputs)} inputs using vLLM")
            
            prompts = []
            for user_input in user_inputs:
                messages = [{"role": "user", "content": user_input}]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking
                )
                prompts.append(prompt)
            
            params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
            )
            
            start_time = time.time()
            outputs = self.llm.generate(prompts, sampling_params=params)
            end_time = time.time()
            
            responses = []
            for i, output in enumerate(outputs):
                output_text = output.outputs[0].text.strip()
                
                response_ids = self.tokenizer(output_text, return_tensors="pt").input_ids[0].tolist()
                
                try:
                    end_thinking_token_id = 151668
                    index = len(response_ids) - response_ids[::-1].index(end_thinking_token_id)
                except ValueError:
                    if self.enable_thinking:
                        logger.debug(f"No </think> found in output {i}")
                    index = 0
                
                thinking_content = self.tokenizer.decode(response_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(response_ids[index:], skip_special_tokens=True).strip("\n")
                
                responses.append(content.strip())
            
            logger.info(f"Batch response generated in {end_time - start_time:.2f}s for {len(user_inputs)} inputs")
            return responses
        
        return self._retry_with_backoff(_generate_batch) or [None] * len(user_inputs)


class LLMService:
    
    def __init__(self, client: BaseLLMClient):
        self.client = client
    
    def get_functionality_from_code(self, code: str) -> Optional[List[str]]:
        user_input = f"""
        Suppose you are a senior software engineer. Your task is to Analyze the following method using the docstring. 
        Extract the functionalities from the docstring on what the method has to offer. And these functionalities must be testable only by calling the 
        method or by using the method in the test code or by doing some other things from the return value of the method. 
        Focus ONLY on the docstring to extract the functionalities. 
        Ensure each functionality is described with enough context to distinguish its different uses and these functionalities must be testable differently only by calling the method. 
        Do not include any functionality that is not testable only by calling the method.
        Do not include duplicate functionalities.
        List the functionalities under the <functionalities><functionality>functionality_1</functionality></functionalities> tag format without any jargon or extra words or numbers.  Here is the function:
        {code}
        """
        
        response = self.client.generate_response(user_input)
        if not response:
            return None
        
        return self._parse_functionalities(response)
    
    def get_tested_functionality_from_test_code(
        self, 
        test_code: str, 
        functionalities: List[str], 
        method_name: str, 
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        user_input = f"""
        Assume the role of a senior software engineer. Your task is to analyze the following test method and determine:

        1. Whether any functionality from the list below (corresponding to the method <method>{method_name}</method>) is being exercised — either directly or indirectly — through one or two hops in the method call graph (e.g., wrapper, helper, or intermediate method calls).
        2. If so, identify the specific functionality (or functionalities) being tested.

        Your output must follow this structure:
        - Use a <method> tag to indicate the method whose functionalities are being evaluated. This must always be the method I provided: <method>{method_name}</method>
        - Use a <tested_functionalities> tag to wrap one or more <functionality> tags, each containing a functionality description being tested.
        Evaluation Rules:
        - For return-value-based functionalities, do not mark them as tested if the return value is ignored OR if it's used in code paths that don't execute during the test OR it's return value isn't used at all.
        - A functionality is only considered tested if the test case:
            - Observes its behavior via assertions, validations, or state inspections, OR
            - Uses its return value or effects in a way that influences the actual execution flow being tested (e.g., stores it, asserts it, makes decisions based on it, or incorporates it into calculations that affect the test outcome). OR
            - The functionality is considered tested if its behavior (when it has no return value) or its return value contributes to the final program state, outcome, or behavior that the test validates. For functions without a return value, it is sufficient that their behavior influences something the test checks, even if not directly asserted. However, if a function returns a value, the test must observe and confirm its effect in some way.
        - Merely calling a method is not sufficient — its effects must be observed or its return value must contribute to the execution path that the test actually exercises.
        - Wrapper/helper methods are valid paths if they invoke the original method within any call levels.
        - If none of the listed functionalities of the provided method are tested in any way, then output:
            <functionality>-1</functionality>
        
        Do not include any other text, explanation, or numbering. Output only the specified XML-like tags.

        Details:
        <method>{method_name}</method>
        Available functionalities:
        {chr(10).join(f"<functionality>{f}</functionality>" for f in functionalities)}

        Test method to analyze:
        {test_code}
        """
        
        if context:
            user_input += f"""
            Additional context:
            The following method implementations may be relevant for understanding indirect interactions or method call chains:
            {context}
            """
        
        response = self.client.generate_response(user_input)
        if not response:
            return None
        
        return self._parse_test_mapping(response)
    
    def get_tested_functionality_from_test_code_batch(
        self,
        test_codes: List[str],
        functionalities: List[str],
        method_name: str,
        contexts: Optional[List[str]] = None
    ) -> List[Optional[Dict[str, Any]]]:
        if contexts is None:
            contexts = [None] * len(test_codes)
        
        if len(test_codes) != len(contexts):
            logger.warning(f"Mismatch: {len(test_codes)} test codes but {len(contexts)} contexts. Using None for missing contexts.")
            contexts = contexts + [None] * (len(test_codes) - len(contexts))
        
        user_inputs = []
        for i, test_code in enumerate(test_codes):
            context = contexts[i] if i < len(contexts) else None
            
            user_input = f"""
        Assume the role of a senior software engineer. Your task is to analyze the following test method and determine:

        1. Whether any functionality from the list below (corresponding to the method <method>{method_name}</method>) is being exercised — either directly or indirectly — through one or two hops in the method call graph (e.g., wrapper, helper, or intermediate method calls).
        2. If so, identify the specific functionality (or functionalities) being tested.

        Your output must follow this structure:
        - Use a <method> tag to indicate the method whose functionalities are being evaluated. This must always be the method I provided: <method>{method_name}</method>
        - Use a <tested_functionalities> tag to wrap one or more <functionality> tags, each containing a functionality description being tested.
        Evaluation Rules:
        - For return-value-based functionalities, do not mark them as tested if the return value is ignored OR if it's used in code paths that don't execute during the test OR it's return value isn't used at all.
        - A functionality is only considered tested if the test case:
            - Observes its behavior via assertions, validations, or state inspections, OR
            - Uses its return value or effects in a way that influences the actual execution flow being tested (e.g., stores it, asserts it, makes decisions based on it, or incorporates it into calculations that affect the test outcome). OR
            - The functionality is considered tested if its behavior (when it has no return value) or its return value contributes to the final program state, outcome, or behavior that the test validates. For functions without a return value, it is sufficient that their behavior influences something the test checks, even if not directly asserted. However, if a function returns a value, the test must observe and confirm its effect in some way.
        - Merely calling a method is not sufficient — its effects must be observed or its return value must contribute to the execution path that the test actually exercises.
        - Wrapper/helper methods are valid paths if they invoke the original method within any call levels.
        - If none of the listed functionalities of the provided method are tested in any way, then output:
            <functionality>-1</functionality>
        
        Do not include any other text, explanation, or numbering. Output only the specified XML-like tags.

        Details:
        <method>{method_name}</method>
        Available functionalities:
        {chr(10).join(f"<functionality>{f}</functionality>" for f in functionalities)}

        Test method to analyze:
        {test_code}
        """
            
            if context:
                user_input += f"""
            Additional context:
            The following method implementations may be relevant for understanding indirect interactions or method call chains:
            {context}
            """
            
            user_inputs.append(user_input)
        
        logger.info(f"Processing batch of {len(test_codes)} test methods for method {method_name}")
        responses = self.client.generate_batch_response(user_inputs)
        
        results = []
        for i, response in enumerate(responses):
            if response is None:
                logger.warning(f"Failed to get response for test method {i}")
                results.append(None)
            else:
                result = self._parse_test_mapping(response)
                results.append(result)
        
        return results
    
    def get_not_tested_functionalities(
        self, 
        functionalities: List[str], 
        tested_functionalities: List[str]
    ) -> Optional[List[str]]:
        functionalities = [f.strip() for f in functionalities]
        tested_functionalities = [f.strip() for f in tested_functionalities]
        return list(set([f for f in functionalities if f not in tested_functionalities]))
    
    def _parse_functionalities(self, xml_string: str) -> Optional[List[str]]:
        try:
            functionality_matches = re.findall(r'<functionality>(.*?)</functionality>', xml_string, re.DOTALL)
            if functionality_matches:
                return [f.strip() for f in functionality_matches]
            
            root = ET.fromstring(xml_string)
            functionalities = [elem.text.strip() if elem.text else "" for elem in root.findall('functionality')]
            return [f for f in functionalities if f]
        except Exception as e:
            logger.error(f"Failed to parse functionalities: {e}")
            logger.debug(f"XML string: {xml_string[:500]}")
            return None
    
    def _parse_test_mapping(self, xml_string: str) -> Optional[Dict[str, Any]]:
        try:
            method_match = re.search(r'<method>(.*?)</method>', xml_string)
            functionality_matches = re.findall(r'<functionality>(.*?)</functionality>', xml_string, re.DOTALL)
            
            if method_match or functionality_matches:
                return {
                    'method': method_match.group(1) if method_match else None,
                    'functionalities': [f.strip() for f in functionality_matches if f.strip()]
                }
            
            root = ET.fromstring(xml_string)
            method = root.find('method').text if root.find('method') is not None else None
            functionalities = [elem.text.strip() if elem.text else "" for elem in root.findall('functionality')]
            return {
                'method': method,
                'functionalities': [f for f in functionalities if f]
            }
        except ET.ParseError as e:
            logger.error(f"Failed to parse test mapping: {e}")
            logger.debug(f"XML string: {xml_string[:500]}")
            return None


def create_llm_client(client_type: str = "qwen", use_vllm: bool = True, **kwargs) -> BaseLLMClient:
    model_name = kwargs.pop('model_name', '/scratch/pppaul/cache_llm_models/Qwen14B')
    enable_thinking = kwargs.pop('enable_thinking', True)
    batch_size = kwargs.pop('batch_size', 2)
    
    if use_vllm:
        return QwenChatbotV2(model_name, enable_thinking, batch_size, **kwargs)
    else:
        return QwenChatbot(model_name, enable_thinking, batch_size, **kwargs)
