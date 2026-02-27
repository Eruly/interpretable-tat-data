from paddleocr import PaddleOCRVL
import os
import traceback

os.makedirs("output", exist_ok=True)

try:
    print("Initializing PaddleOCRVL with remote server...")
    # Inspecting the code, we might want to catch the DependencyError directly to see what's missing
    from paddlex.utils.deps import DependencyError
    from paddlex import create_pipeline
    
    pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://localhost:8080/v1")
    print("Running prediction...")
    output = pipeline.predict("/home/csle/DATA/dkyoon/interpretable-llm/table_interpretable/test_render.png")
    
    for res in output:
        res.print()
        res.save_to_json(save_path="output")
        res.save_to_markdown(save_path="output")
    print("Test completed successfully.")

except Exception as e:
    print(f"Test failed with error: {e}")
    # try to reveal the original exception cause
    if hasattr(e, '__cause__') and e.__cause__:
        print(f"Caused by: {e.__cause__}")
        traceback.print_exception(type(e.__cause__), e.__cause__, e.__cause__.__traceback__)
    else:
        traceback.print_exc()
