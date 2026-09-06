import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# Add the root directory to the system path so we can import our app files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from create_memory_for_llm import create_chunks, build_vector_store
from MedibotUI import set_custom_prompt, load_llm

def test_create_chunks():
    """
    Test that the chunking logic splits large text and preserves source metadata.
    """
    # 1. Arrange: Create a fake document with 1000 'A' characters
    long_text = "A" * 1000
    mock_document = Document(
        page_content=long_text, 
        metadata={"source": "fake_medical_book.pdf", "page": 5}
    )
    
    # 2. Act: Run the chunker
    chunks = create_chunks([mock_document], chunk_size=500, chunk_overlap=50)
    
    # 3. Assert: Verify the output
    assert len(chunks) > 1, "The document should be split into multiple chunks"
    assert chunks[0].metadata["source"] == "fake_medical_book.pdf", "Metadata source must be preserved"
    assert chunks[0].metadata["page"] == 5, "Metadata page number must be preserved"
    assert len(chunks[0].page_content) <= 500, "Chunk size should respect the 500 character limit"

def test_set_custom_prompt():
    """
    Test that the custom prompt includes the necessary instruction safeguards.
    """
    prompt = set_custom_prompt()
    
    # The prompt MUST require these two specific inputs for LangChain RetrievalQA to work
    assert "context" in prompt.input_variables
    assert "question" in prompt.input_variables
    
    # The prompt MUST contain our strict fallback instruction
    assert "reply exactly: I don't know" in prompt.template

@patch('create_memory_for_llm.load_pdfs')
@patch('create_memory_for_llm.os.makedirs')
def test_build_vector_store_empty_directory(mock_makedirs, mock_load_pdfs):
    """
    Test that if no PDFs are uploaded, the vector store builder aborts gracefully.
    """
    # 1. Arrange: Force load_pdfs to return an empty list
    mock_load_pdfs.return_value = []
    
    # 2. Act: Call the function
    result = build_vector_store()
    
    # 3. Assert: It should return False
    assert result is False

# CHANGED: We now patch ChatHuggingFace as well to bypass LangChain's strict type validation
@patch('MedibotUI.ChatHuggingFace')
@patch('MedibotUI.HuggingFaceEndpoint')
def test_load_llm_configuration(mock_hf_endpoint, mock_chat_hf):
    """
    Test that the LLM is initialized with the correct model and parameters.
    """
    # 1. Act: Initialize the LLM
    llm = load_llm()
    
    # 2. Assert: Verify HuggingFaceEndpoint was called with our specific Qwen model
    mock_hf_endpoint.assert_called_once()
    
    # Extract the arguments passed to HuggingFaceEndpoint
    _, kwargs = mock_hf_endpoint.call_args
    
    assert kwargs['repo_id'] == "Qwen/Qwen2.5-1.5B-Instruct", "Wrong model repository specified"
    assert kwargs['temperature'] == 0.5, "Temperature should be set to 0.5 for stable answers"
    
    # 3. Assert: Verify ChatHuggingFace was also called wrapper around the endpoint
    mock_chat_hf.assert_called_once_with(llm=mock_hf_endpoint.return_value)