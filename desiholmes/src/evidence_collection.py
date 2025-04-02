# src/evidence_collection.py
import os
import logging
from typing import List, Dict, Any
import spacy
from PIL import Image
import pytesseract
from transformers import pipeline

from src.models import Evidence, EvidenceItem

class EvidenceCollector:
    """Collects and processes evidence from various sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_md")
            self.logger.info("Loaded spaCy model")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
        
        try:
            self.image_captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
            self.logger.info("Loaded image captioning model")
        except Exception as e:
            self.logger.error(f"Failed to load image captioning model: {e}")
            self.image_captioner = None
    
    def collect(self, evidence_dir: str) -> Evidence:
        """
        Collect evidence from the specified directory.
        
        Args:
            evidence_dir: Directory containing evidence files
            
        Returns:
            Evidence object containing processed evidence items
        """
        self.logger.info(f"Collecting evidence from {evidence_dir}")
        evidence = Evidence()
        
        # Process each file in the evidence directory
        for root, _, files in os.walk(evidence_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                self.logger.info(f"Processing file: {file_path}")
                
                # Determine file type and process accordingly
                file_ext = os.path.splitext(filename)[1].lower()
                
                try:
                    if file_ext in ['.txt', '.md', '.doc', '.docx', '.pdf']:
                        item = self._process_text_file(file_path)
                    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        item = self._process_image_file(file_path)
                    else:
                        self.logger.warning(f"Unsupported file type: {file_ext}")
                        continue
                    
                    if item:
                        evidence.add_item(item)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
        
        self.logger.info(f"Collected {len(evidence.items)} evidence items")
        return evidence
    
    def _process_text_file(self, file_path: str) -> EvidenceItem:
        """Process a text file to extract content and metadata."""
        try:
            # For simplicity, we're just reading text files
            # In a real implementation, you'd use libraries like textract
            # to handle various document formats
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract entities using spaCy
            entities = self._extract_entities(text)
            
            # Create evidence item
            item = EvidenceItem(
                type="text",
                source=file_path,
                content=text,
                metadata={
                    "filename": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path),
                    "entities": entities
                }
            )
            
            return item
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {e}")
            return None
    
    def _process_image_file(self, file_path: str) -> EvidenceItem:
    #"""Process an image file to extract content and metadata."""
    try:
        # Load image
        img = Image.open(file_path)
        
        # Extract image metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "dimensions": img.size,
            "format": img.format,
            "mode": img.mode
        }
        
        # Extract text from image using OCR
        try:
            extracted_text = pytesseract.image_to_string(img)
            metadata["extracted_text"] = extracted_text
            
            # Extract entities from the OCR text
            if extracted_text and self.nlp:
                metadata["entities"] = self._extract_entities(extracted_text)
        except Exception as e:
            self.logger.warning(f"OCR failed for {file_path}: {e}")
        
        # Generate image caption
        try:
            if self.image_captioner:
                caption = self.image_captioner(file_path)[0]['generated_text']
                metadata["caption"] = caption
        except Exception as e:
            self.logger.warning(f"Image captioning failed for {file_path}: {e}")
        
        # Create evidence item
        item = EvidenceItem(
            type="image",
            source=file_path,
            content=file_path,  # Store the path to the image
            metadata=metadata
        )
        
        return item
    except Exception as e:
        self.logger.error(f"Error processing image file {file_path}: {e}")
        return None

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using spaCy."""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
    return entities
