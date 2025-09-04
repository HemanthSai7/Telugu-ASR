import json
import time
import os
from openai import OpenAI
from typing import Dict, List, Any
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeluguTransliterationBatch:
    def __init__(self, api_key: str = None):
        """
        Initialize the batch processor for Telugu transliteration
        
        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key)
        
    def prepare_batch_requests(self, input_jsonl_path: str, output_batch_path: str) -> str:
        """
        Prepare batch requests from input JSONL file
        
        Args:
            input_jsonl_path: Path to input JSONL file with Telugu texts
            output_batch_path: Path to save batch requests file
            
        Returns:
            Path to the batch requests file
        """
        batch_requests = []
        
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                
                # Extract Telugu text (adjust field name based on your JSONL structure)
                telugu_text = data.get('telugu_text') or data.get('text') or data.get('transcript')
                
                if not telugu_text:
                    logger.warning(f"No Telugu text found in line {i+1}, skipping...")
                    continue
                
                # Create batch request
                request = {
                    "custom_id": f"request_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",  # Cost-effective model for transliteration
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a Telugu to English transliteration expert. Transliterate the given Telugu text to English using only standard Latin characters (a-z, A-Z) without diacritics or special symbols. Maintain the phonetic pronunciation as closely as possible. Only return the transliterated text, no explanations."
                            },
                            {
                                "role": "user",
                                "content": f"Transliterate this Telugu text to English: {telugu_text}"
                            }
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.1  # Low temperature for consistency
                    }
                }
                
                # Store original data for later merging
                request["original_data"] = {
                    "wav_path": data.get('wav_path', ''),
                    "duration": data.get('duration', ''),
                    "telugu_text": telugu_text,
                    "line_index": i
                }
                
                batch_requests.append(request)
        
        # Save batch requests to file
        with open(output_batch_path, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                # Remove original_data before saving (OpenAI doesn't accept extra fields)
                clean_request = {k: v for k, v in request.items() if k != "original_data"}
                f.write(json.dumps(clean_request) + '\n')
        
        logger.info(f"Prepared {len(batch_requests)} batch requests")
        return output_batch_path
    
    def upload_and_create_batch(self, batch_file_path: str) -> str:
        """
        Upload batch file and create batch job
        
        Args:
            batch_file_path: Path to batch requests file
            
        Returns:
            Batch job ID
        """
        # Upload the batch file
        with open(batch_file_path, 'rb') as f:
            batch_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        logger.info(f"Uploaded batch file: {batch_file.id}")
        
        # Create batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        logger.info(f"Created batch job: {batch_job.id}")
        return batch_job.id
    
    def wait_for_completion(self, batch_id: str, check_interval: int = 60) -> Dict[str, Any]:
        """
        Wait for batch job completion
        
        Args:
            batch_id: Batch job ID
            check_interval: Check interval in seconds
            
        Returns:
            Completed batch job details
        """
        logger.info(f"Waiting for batch {batch_id} to complete...")
        
        while True:
            batch_job = self.client.batches.retrieve(batch_id)
            
            if batch_job.status == "completed":
                logger.info(f"Batch {batch_id} completed successfully!")
                return batch_job
            elif batch_job.status == "failed":
                raise Exception(f"Batch {batch_id} failed: {batch_job.errors}")
            elif batch_job.status == "cancelled":
                raise Exception(f"Batch {batch_id} was cancelled")
            
            logger.info(f"Batch status: {batch_job.status}. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)
    
    def download_and_process_results(self, batch_job: Dict[str, Any], 
                                   original_jsonl_path: str, 
                                   output_jsonl_path: str) -> None:
        """
        Download batch results and merge with original data
        
        Args:
            batch_job: Completed batch job details
            original_jsonl_path: Path to original JSONL file
            output_jsonl_path: Path to save final results
        """
        # Download results
        result_file_id = batch_job.output_file_id
        result = self.client.files.content(result_file_id)
        
        # Parse results
        results_data = {}
        for line in result.text.split('\n'):
            if line.strip():
                result_item = json.loads(line)
                custom_id = result_item['custom_id']
                
                if result_item.get('response') and result_item['response'].get('body'):
                    english_text = result_item['response']['body']['choices'][0]['message']['content'].strip()
                    results_data[custom_id] = english_text
                else:
                    # Handle errors
                    error_msg = result_item.get('error', 'Unknown error')
                    logger.warning(f"Error for {custom_id}: {error_msg}")
                    results_data[custom_id] = ""
        
        # Merge with original data and create final output
        final_results = []
        
        with open(original_jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                original_data = json.loads(line.strip())
                custom_id = f"request_{i}"
                
                # Get transliterated text
                english_text = results_data.get(custom_id, "")
                
                # Extract Telugu text (same logic as before)
                telugu_text = original_data.get('telugu_text') or original_data.get('text') or original_data.get('transcript')
                
                if telugu_text:  # Only include if we had Telugu text
                    final_result = {
                        "wav_path": original_data.get('wav_path', ''),
                        "duration": original_data.get('duration', ''),
                        "telugu_text": telugu_text,
                        "english_text": english_text
                    }
                    final_results.append(final_result)
        
        # Save final results
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for result in final_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(final_results)} transliterated results to {output_jsonl_path}")
    
    def process_file(self, input_jsonl_path: str, output_jsonl_path: str, 
                    temp_dir: str = "temp_batch_files") -> None:
        """
        Complete pipeline to process a JSONL file
        
        Args:
            input_jsonl_path: Path to input JSONL file
            output_jsonl_path: Path to save final results
            temp_dir: Directory for temporary files
        """
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Step 1: Prepare batch requests
            batch_file_path = os.path.join(temp_dir, "batch_requests.jsonl")
            self.prepare_batch_requests(input_jsonl_path, batch_file_path)
            
            # Step 2: Upload and create batch
            batch_id = self.upload_and_create_batch(batch_file_path)
            
            # Step 3: Wait for completion
            completed_batch = self.wait_for_completion(batch_id)
            
            # Step 4: Download and process results
            self.download_and_process_results(completed_batch, input_jsonl_path, output_jsonl_path)
            
        finally:
            # Cleanup temp files
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)


def main():
    """
    Example usage of the Telugu transliteration batch processor
    """
    # Initialize the processor
    processor = TeluguTransliterationBatch(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Set your file paths
    input_file = "/home/hemanth/GIT_Projects/RA/data/IISc_RESPIN_dev_te/meta_dev_te.jsonl"
    output_file = "/home/hemanth/GIT_Projects/RA/data/IISc_RESPIN_dev_te/transliterated_results.jsonl"

    # Process the file
    try:
        processor.process_file(input_file, output_file)
        print(f"Transliteration completed! Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")


if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    main()