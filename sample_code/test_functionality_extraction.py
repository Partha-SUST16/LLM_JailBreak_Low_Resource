import json
import re
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from llm_client import create_llm_client, LLMService
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_functionality_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MethodInfo:
    method_name: str
    class_name: str
    package_name: str
    parameters: str
    code: str
    javadoc: str
    test_methods: List[Dict[str, Any]]


@dataclass
class ProcessingResult:
    method_under_test: str
    class_name: str
    functionalities: List[str]
    not_tested_functionalities: List[str]
    tested_functionalities: List[str]
    called_test_methods: List[Dict[str, Any]]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class JsonConverter:
    
    @staticmethod
    def convert_new_structure_to_old_format(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            candidate_method = data.get("candidateMethod", "")
            if not candidate_method:
                logger.error("Missing candidateMethod field")
                return None
            
            paren_pos = candidate_method.find("(")
            if paren_pos == -1:
                logger.error(f"Invalid candidateMethod format: {candidate_method}")
                return None
            
            method_part = candidate_method[:paren_pos]
            last_dot_pos = method_part.rfind(".")
            if last_dot_pos == -1:
                logger.error(f"Invalid candidateMethod format: {candidate_method}")
                return None
            
            full_class_name = candidate_method[:last_dot_pos]
            method_name = candidate_method[last_dot_pos + 1:paren_pos]
            method_signature = candidate_method[paren_pos:]
            
            if not method_name or not full_class_name:
                logger.error(f"Invalid candidateMethod: {candidate_method}")
                return None
            
            candidate_source_code = ""
            candidate_javadoc = ""
            
            test_methods = data.get("testMethods", [])
            for test_method in test_methods:
                shortest_path = test_method.get("shortestPath", [])
                candidate_info = JsonConverter._extract_candidate_from_shortest_path(
                    shortest_path, candidate_method
                )
                if candidate_info:
                    candidate_source_code = candidate_info.get("sourceCode", "")
                    candidate_javadoc = candidate_info.get("javadoc", "")
                    break
            
            if not candidate_source_code:
                logger.warning(f"Could not find candidate method implementation for {candidate_method}")
                for test_method in test_methods:
                    shortest_path = test_method.get("shortestPath", [])
                    for path_node in shortest_path:
                        if path_node.get("id") == candidate_method:
                            candidate_source_code = path_node.get("sourceCode", "")
                            candidate_javadoc = path_node.get("javadoc", "")
                            break
                        for child in path_node.get("children", []):
                            if child.get("id") == candidate_method:
                                candidate_source_code = child.get("sourceCode", "")
                                candidate_javadoc = child.get("javadoc", "")
                                break
                        if candidate_source_code:
                            break
                    if candidate_source_code:
                        break
            
            called_by_test_methods = []
            seen_test_method_ids = set()
            
            for test_method in test_methods:
                test_method_id = test_method.get("id", "")
                if not test_method_id:
                    test_class_name = test_method.get("className", "")
                    test_method_name = test_method.get("methodName", "")
                    test_method_sig = test_method.get("methodSignature", "")
                    test_method_id = f"{test_class_name}.{test_method_name}{test_method_sig}"
                
                if test_method_id in seen_test_method_ids:
                    continue
                
                seen_test_method_ids.add(test_method_id)
                
                test_method_name = test_method.get("methodName", "")
                test_class_name = test_method.get("className", "")
                test_source_code = test_method.get("sourceCode", "")
                shortest_path = test_method.get("shortestPath", [])
                
                if not test_source_code or test_source_code.strip() == "":
                    if shortest_path and len(shortest_path) > 0:
                        first_node = shortest_path[0]
                        first_node_source = first_node.get("sourceCode", "")
                        if "@Test" in first_node_source or "@ParameterizedTest" in first_node_source:
                            test_source_code = first_node_source
                
                called_methods = JsonConverter._convert_shortest_path_to_called_methods(
                    shortest_path, exclude_first_if_test=True
                )
                
                called_by_test_methods.append({
                    "testMethod": f"{test_class_name}.{test_method_name}",
                    "testMethodSourceCode": test_source_code,
                    "calledMethods": called_methods
                })
            
            converted_data = {
                "methodName": method_name,
                "className": full_class_name,
                "methodSignature": method_signature,
                "sourceCode": candidate_source_code,
                "javaDoc": candidate_javadoc,
                "calledByTestMethods": called_by_test_methods
            }
            
            return converted_data
            
        except Exception as e:
            logger.error(f"Error converting new structure to old format: {e}")
            return None
    
    @staticmethod
    def _extract_candidate_from_shortest_path(
        shortest_path: List[Dict[str, Any]], 
        candidate_method_id: str
    ) -> Optional[Dict[str, Any]]:
        def search_recursively(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if node.get("id") == candidate_method_id:
                return {
                    "sourceCode": node.get("sourceCode", ""),
                    "javadoc": node.get("javadoc", "")
                }
            
            for child in node.get("children", []):
                result = search_recursively(child)
                if result:
                    return result
            return None
        
        for node in shortest_path:
            result = search_recursively(node)
            if result:
                return result
        return None
    
    @staticmethod
    def _convert_shortest_path_to_called_methods(
        shortest_path: List[Dict[str, Any]], 
        exclude_first_if_test: bool = True
    ) -> List[Dict[str, Any]]:
        called_methods = []
        
        if not shortest_path:
            return called_methods
        
        start_idx = 0
        if exclude_first_if_test and len(shortest_path) > 0:
            first_node = shortest_path[0]
            first_source = first_node.get("sourceCode", "")
            if "@Test" in first_source or "@ParameterizedTest" in first_source:
                start_idx = 1
        
        def extract_all_methods(node: Dict[str, Any], methods: List[Dict[str, Any]]):
            method_id = node.get("id", "")
            if method_id:
                methods.append({
                    "fullMethodName": method_id,
                    "className": node.get("className", ""),
                    "methodName": node.get("methodName", ""),
                    "methodSignature": node.get("methodSignature", ""),
                    "sourceCode": node.get("sourceCode", "")
                })
            
            for child in node.get("children", []):
                extract_all_methods(child, methods)
        
        for i in range(start_idx, len(shortest_path)):
            extract_all_methods(shortest_path[i], called_methods)
        
        return called_methods


class FunctionalityProcessor:
    
    def __init__(self, llm_service: LLMService, batch_size: int = 2):
        self.llm_service = llm_service
        self.batch_size = batch_size
    
    def process_method(self, item: Dict[str, Any]) -> Optional[ProcessingResult]:
        start_time = time.time()
        
        try:
            method_info = self._extract_method_info(item)
            if not method_info:
                return self._create_error_result(
                    "Failed to extract method information", 
                    start_time, 
                    time.time()
                )
            
            logger.info(f"Processing method: {method_info.class_name}.{method_info.method_name}")
            logger.info(f"Number of test methods: {len(method_info.test_methods)}")
            
            if not self._validate_method_data(method_info):
                return self._create_error_result(
                    f"Invalid method data for {method_info.method_name}", 
                    start_time, 
                    time.time()
                )
            
            logger.info("Extracting functionalities from method...")
            functionalities = self._extract_functionalities(method_info)
            if not functionalities:
                return self._create_error_result(
                    f"Failed to extract functionalities for {method_info.method_name}", 
                    start_time, 
                    time.time()
                )
            
            logger.info(f"Extracted {len(functionalities)} functionalities")
            
            logger.info("Analyzing test methods...")
            test_analysis = self._analyze_test_methods(method_info, functionalities)
            
            not_tested_functionalities = self._get_untested_functionalities(
                functionalities, 
                test_analysis['tested_functionalities']
            )
            
            if not_tested_functionalities is None:
                not_tested_functionalities = ["<ERROR>Could not extract not tested functionalities</ERROR>"]
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"Processing completed in {processing_time:.2f}s")
            logger.info(f"Tested functionalities: {len(test_analysis['tested_functionalities'])}")
            logger.info(f"Untested functionalities: {len(not_tested_functionalities)}")
            
            return ProcessingResult(
                method_under_test=method_info.method_name,
                class_name=method_info.class_name,
                functionalities=functionalities,
                not_tested_functionalities=not_tested_functionalities,
                tested_functionalities=test_analysis['tested_functionalities'],
                called_test_methods=test_analysis['called_test_methods'],
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing method: {e}", exc_info=True)
            return self._create_error_result(str(e), start_time, time.time())
    
    def _extract_method_info(self, item: Dict[str, Any]) -> Optional[MethodInfo]:
        try:
            class_name = item["className"].split(".")[-1]
            parameters = item["methodSignature"].replace("(", "").replace(")", "")
            package_name = ".".join(item["className"].split(".")[:-1])
            code = item["sourceCode"]
            code = re.sub(r"/\*\*[\s\S]*?\*/", "", code)
            javadoc = item.get("javaDoc", "")
            test_methods = item["calledByTestMethods"]
            
            return MethodInfo(
                method_name=item["methodName"],
                class_name=class_name,
                package_name=package_name,
                parameters=parameters,
                code=code,
                javadoc=javadoc,
                test_methods=test_methods
            )
        except KeyError as e:
            logger.error(f"Missing required field in method data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting method info: {e}")
            return None
    
    def _validate_method_data(self, method_info: MethodInfo) -> bool:
        if not method_info.javadoc:
            logger.warning(f"No javadoc for {method_info.method_name} {method_info.class_name}")
            return False
        
        if not method_info.javadoc.strip() or not method_info.code.strip():
            logger.warning(f"Empty javadoc or code for {method_info.method_name} {method_info.class_name}")
            return False
        
        return True
    
    def _extract_functionalities(self, method_info: MethodInfo) -> Optional[List[str]]:
        code_with_javadoc = f"{method_info.javadoc}\n{method_info.code}"
        return self.llm_service.get_functionality_from_code(code_with_javadoc)
    
    def _analyze_test_methods(
        self, 
        method_info: MethodInfo, 
        functionalities: List[str]
    ) -> Dict[str, Any]:
        tested_functionalities = []
        called_test_methods = []
        
        test_start_time = time.time()
        
        if not method_info.test_methods or len(method_info.test_methods) == 0:
            logger.info("No test methods found")
            return {
                'tested_functionalities': [],
                'called_test_methods': []
            }
        
        test_codes = []
        contexts = []
        test_method_names = []
        
        for test_method in method_info.test_methods:
            test_code = test_method.get("testMethodSourceCode", "")
            if not test_code.strip():
                continue
            
            context = test_method.get("calledMethods", [])
            context_str = self._build_test_context(context, method_info)
            
            test_codes.append(test_code)
            contexts.append(context_str if context_str else None)
            test_method_names.append(test_method.get("testMethod", ""))
        
        if not test_codes:
            logger.warning("No valid test method codes found")
            return {
                'tested_functionalities': [],
                'called_test_methods': []
            }
        
        method_identifier = f"{method_info.class_name}.{method_info.method_name}"
        all_results = []
        
        num_batches = (len(test_codes) + self.batch_size - 1) // self.batch_size
        logger.info(f"Processing {len(test_codes)} test methods in {num_batches} batches of size {self.batch_size}")
        
        for batch_idx in range(0, len(test_codes), self.batch_size):
            batch_end = min(batch_idx + self.batch_size, len(test_codes))
            batch_codes = test_codes[batch_idx:batch_end]
            batch_contexts = contexts[batch_idx:batch_end] if contexts else [None] * len(batch_codes)
            
            logger.info(f"Processing batch {batch_idx // self.batch_size + 1}/{num_batches} ({len(batch_codes)} test methods)")
            
            batch_results = self.llm_service.get_tested_functionality_from_test_code_batch(
                batch_codes,
                functionalities,
                method_identifier,
                batch_contexts
            )
            
            all_results.extend(batch_results)
        
        for i, result in enumerate(all_results):
            test_method_name = test_method_names[i] if i < len(test_method_names) else f"test_method_{i}"
            
            if result is None:
                logger.warning(f"Failed to get tested functionalities for {test_method_name}")
                called_test_methods.append({
                    "methodName": test_method_name,
                    "functionalities": []
                })
                continue
            
            tested_method = result.get("method")
            tested_functionalities_in_method = result.get("functionalities", [])
            
            if tested_method == method_identifier:
                tested_functionalities.extend(tested_functionalities_in_method)
            else:
                logger.warning(f"Tested method {tested_method} does not match {method_identifier}")
            
            called_test_methods.append({
                "methodName": test_method_name,
                "functionalities": tested_functionalities_in_method
            })
        
        test_time = time.time() - test_start_time
        logger.info(f"Test method analysis completed in {test_time:.2f}s")
        
        return {
            'tested_functionalities': tested_functionalities,
            'called_test_methods': called_test_methods
        }
    
    def _build_test_context(
        self, 
        context: List[Dict[str, Any]], 
        method_info: MethodInfo
    ) -> str:
        if not context:
            return ""
        
        context_str = ""
        already_added_methods = []
        
        for c in context:
            method_key = c.get("fullMethodName", "")
            if method_key not in already_added_methods:
                class_name = c.get("className", "").split(".")[-1]
                source_code = c.get("sourceCode", "")
                source_code = re.sub(r"/\*\*[\s\S]*?\*/", "", source_code)
                source_code = f"{class_name} {source_code}"
                context_str += f"{source_code}\n"
                already_added_methods.append(method_key)
        
        return context_str
    
    def _get_untested_functionalities(
        self, 
        functionalities: List[str], 
        tested_functionalities: List[str]
    ) -> Optional[List[str]]:
        return self.llm_service.get_not_tested_functionalities(functionalities, tested_functionalities)
    
    def _create_error_result(
        self, 
        error_message: str, 
        start_time: float, 
        end_time: float
    ) -> ProcessingResult:
        return ProcessingResult(
            method_under_test="unknown",
            class_name="unknown",
            functionalities=[],
            not_tested_functionalities=[],
            tested_functionalities=[],
            called_test_methods=[],
            processing_time=end_time - start_time,
            success=False,
            error_message=error_message
        )


def load_json_files(data_dir: Path) -> List[Dict[str, Any]]:
    json_files = list(data_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")
    
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
                logger.info(f"Loaded {json_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
    
    return all_data


def save_result(result: ProcessingResult, output_dir: Path, parameters: str = ""):
    try:
        clean_parameters = re.sub(r'[^a-zA-Z0-9]', '', parameters)
        filename = f"{result.class_name}_{result.method_under_test}"
        if clean_parameters:
            filename += f"_{clean_parameters}"
        
        if not result.success:
            filename += "_error"
        
        filename += ".json"
        file_path = output_dir / filename
        
        data = asdict(result)
        
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved result to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save result: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test functionality extraction')
    parser.add_argument('--model-name', type=str, default='/scratch/pppaul/cache_llm_models/Qwen14B',
                        help='Path to the model')
    parser.add_argument('--use-vllm', action='store_true', default=False, help='Use vllm instead of AutoModelForCausalLM')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for LLM processing')
    parser.add_argument('--enable-thinking', action='store_true', default=True, help='Enable thinking mode')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing input JSON files')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for output JSON files')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("Starting Functionality Extraction Test")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Using vllm: {args.use_vllm}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    logger.info("Initializing LLM client...")
    llm_client = create_llm_client(
        client_type="qwen",
        use_vllm=args.use_vllm,
        model_name=args.model_name,
        batch_size=args.batch_size,
        enable_thinking=args.enable_thinking
    )
    llm_service = LLMService(llm_client)
    
    processor = FunctionalityProcessor(llm_service, batch_size=args.batch_size)
    
    logger.info("Loading JSON files...")
    json_data_list = load_json_files(data_dir)
    
    if not json_data_list:
        logger.error("No JSON files found to process")
        return
    
    total_start_time = time.time()
    stats = {
        'processed': 0,
        'total': len(json_data_list),
        'successes': 0,
        'errors': 0
    }
    
    for i, json_data in enumerate(json_data_list):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file {i+1}/{len(json_data_list)}")
        logger.info(f"{'='*60}")
        
        converted_data = JsonConverter.convert_new_structure_to_old_format(json_data)
        if not converted_data:
            logger.error("Failed to convert JSON structure")
            stats['errors'] += 1
            continue
        
        result = processor.process_method(converted_data)
        if result:
            if result.success:
                stats['successes'] += 1
                save_result(result, output_dir, converted_data.get("methodSignature", ""))
            else:
                stats['errors'] += 1
                logger.error(f"Processing failed: {result.error_message}")
                save_result(result, output_dir, converted_data.get("methodSignature", ""))
        else:
            stats['errors'] += 1
            logger.error("Processing returned None")
        
        stats['processed'] += 1
    
    total_time = time.time() - total_start_time
    
    logger.info("\n" + "="*60)
    logger.info("FUNCTIONALITY EXTRACTION COMPLETED")
    logger.info("="*60)
    logger.info(f"Total files processed: {stats['processed']}")
    logger.info(f"Total files found: {stats['total']}")
    logger.info(f"Successful extractions: {stats['successes']}")
    logger.info(f"Failed extractions: {stats['errors']}")
    logger.info(f"Success rate: {(stats['successes'] / stats['total'] * 100) if stats['total'] > 0 else 0:.1f}%")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info("="*60)


if __name__ == "__main__":
    main()
