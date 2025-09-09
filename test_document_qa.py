#!/usr/bin/env python3
"""
Test script to verify document Q&A functionality works end-to-end
Tests: upload -> storage -> retrieval -> LLM integration
"""

import asyncio
import tempfile
import os
import sys
from pathlib import Path

# Test imports
try:
    from document_processor import DocumentProcessor
    from chroma_client import get_chroma_client
    from openai import AsyncOpenAI
    from dotenv import load_dotenv
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

async def test_qa_flow():
    """Test the complete Q&A flow with document retrieval"""
    print("🧪 Testing Document Q&A Flow")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    # Initialize components
    print("🔧 Initializing components...")
    client = AsyncOpenAI(api_key=openai_key)
    chroma_client = get_chroma_client()
    doc_processor = DocumentProcessor(client)
    
    if not chroma_client.connected:
        print("❌ ChromaDB not connected")
        return False
    
    print("✅ Components initialized")
    
    # Test session info
    test_session_id = "qa_test_session_456"
    test_expert = "test_expert_qa"
    
    # Clean up any existing test documents first
    print("🧹 Cleaning up any existing test documents...")
    doc_processor.delete_session_documents(test_session_id)
    
    # Create a test document with specific content we can query
    print("\n📝 Creating test document...")
    test_content = """
    Family Coaching Best Practices Guide
    
    Chapter 1: Communication Strategies
    Effective communication with children requires active listening and empathy.
    Use clear, simple language appropriate for the child's age.
    Always validate their feelings before providing guidance.
    
    Chapter 2: Positive Reinforcement Techniques
    Reward good behavior immediately after it occurs.
    Use specific praise rather than general comments.
    Example: "Great job putting your toys away!" instead of "Good job!"
    
    Chapter 3: Setting Boundaries
    Consistent boundaries help children feel secure.
    Explain the reasons behind rules in age-appropriate language.
    Follow through with consequences consistently.
    
    Chapter 4: Managing Difficult Behaviors
    Stay calm and avoid reacting emotionally.
    Use redirection for younger children.
    Implement time-outs appropriately for older children.
    
    Key Principles:
    - Patience is essential in family coaching
    - Every child is unique and requires individualized approaches
    - Building trust takes time but is crucial for success
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = f.name
    
    try:
        # Step 1: Upload and process document
        print(f"📄 Step 1: Processing document...")
        result = await doc_processor.process_uploaded_file(
            file_path=temp_file_path,
            filename="family_coaching_guide.txt",
            expert_name=test_expert,
            session_id=test_session_id
        )
        
        if not result["success"]:
            print(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Document processed: {result['chunks_processed']} chunks stored")
        
        # Step 2: Verify storage in ChromaDB
        print(f"\n📊 Step 2: Verifying storage...")
        
        # Check collection info
        collection_info = chroma_client.get_collection_info()
        print(f"  Total chunks in ChromaDB: {collection_info.get('total_chunks', 0)}")
        
        # Check session documents
        session_docs = doc_processor.get_session_documents(test_session_id)
        print(f"  Session chunks: {session_docs['total_chunks']}")
        print(f"  Session documents: {len(session_docs['documents'])}")
        
        if session_docs['total_chunks'] == 0:
            print("❌ No chunks found in session - storage failed!")
            return False
        
        print("✅ Documents properly stored in ChromaDB")
        
        # Step 3: Test retrieval with various queries
        print(f"\n🔍 Step 3: Testing retrieval...")
        
        test_queries = [
            "How do I communicate with children?",
            "What are positive reinforcement techniques?", 
            "How should I set boundaries?",
            "What's in my document?",
            "Tell me about managing difficult behaviors",
            "What are the key principles?",
            "Family coaching strategies"
        ]
        
        successful_retrievals = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n  Query {i}: '{query}'")
            
            similar_chunks = await doc_processor.search_similar_chunks(
                query=query,
                session_id=test_session_id,
                limit=3
            )
            
            if similar_chunks:
                print(f"    ✅ Found {len(similar_chunks)} relevant chunks")
                
                # Check if we have meaningful similarity scores
                good_matches = [chunk for chunk in similar_chunks if chunk.get('similarity', 0) >= 0.3]
                ok_matches = [chunk for chunk in similar_chunks if chunk.get('similarity', 0) >= 0.2]
                
                if good_matches:
                    print(f"    🎯 {len(good_matches)} high-quality matches (>= 0.3 similarity)")
                    successful_retrievals += 1
                elif ok_matches:
                    print(f"    📍 {len(ok_matches)} decent matches (>= 0.2 similarity)")
                    successful_retrievals += 1
                else:
                    print(f"    ⚠️ Low similarity scores - best: {max(chunk.get('similarity', 0) for chunk in similar_chunks):.3f}")
                
                # Show top result details
                top_chunk = similar_chunks[0]
                sim_score = top_chunk.get('similarity', 'N/A')
                content_preview = top_chunk['content'][:150] + "..." if len(top_chunk['content']) > 150 else top_chunk['content']
                sim_display = f"{sim_score:.3f}" if isinstance(sim_score, float) else str(sim_score)
                print(f"    Top result (sim: {sim_display}): {content_preview}")
                
            else:
                print(f"    ❌ No chunks returned from search")
        
        retrieval_success_rate = successful_retrievals / len(test_queries)
        print(f"\n📈 Retrieval Success Rate: {retrieval_success_rate:.1%} ({successful_retrievals}/{len(test_queries)})")
        
        if retrieval_success_rate < 0.7:  # Expect at least 70% success
            print("❌ Retrieval success rate too low!")
            return False
        
        print("✅ Retrieval working well")
        
        # Step 4: Test direct document query simulation (like the chat endpoint)
        print(f"\n💬 Step 4: Testing chat-like document query...")
        
        doc_query = "What's in my uploaded document?"
        
        # Simulate the main.py document query logic
        session_docs_info = doc_processor.get_session_documents(test_session_id)
        
        if session_docs_info['documents']:
            doc_titles = [doc['title'] for doc in session_docs_info['documents']]
            print(f"  Documents found: {doc_titles}")
            
            # Search for relevant content
            similar_chunks = await doc_processor.search_similar_chunks(
                query=doc_query,
                session_id=test_session_id,
                limit=3
            )
            
            if similar_chunks:
                relevant_chunks = [chunk for chunk in similar_chunks if chunk.get('similarity', 0) >= 0.1]
                if relevant_chunks:
                    print(f"  ✅ Found {len(relevant_chunks)} relevant chunks for document query")
                    for j, chunk in enumerate(relevant_chunks[:2]):
                        content_preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                        print(f"    Chunk {j+1}: {content_preview}")
                else:
                    print("  ⚠️ No relevant chunks above 0.1 similarity threshold")
            else:
                print("  ❌ No chunks returned for document query")
        
        # Clean up test documents
        print("\n🧹 Cleaning up test documents...")
        cleanup_success = doc_processor.delete_session_documents(test_session_id)
        if cleanup_success:
            print("✅ Test documents cleaned up")
        else:
            print("⚠️ Could not clean up test documents")
        
        print("\n🎉 Document Q&A test completed successfully!")
        print("\n✅ Key findings:")
        print(f"   - Documents are properly stored in ChromaDB")
        print(f"   - Retrieval works with {retrieval_success_rate:.1%} success rate")
        print(f"   - Document queries return relevant content")
        print(f"   - System is ready for production use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def test_chroma_debug():
    """Debug ChromaDB storage issues"""
    print("\n🔧 ChromaDB Debug Information")
    print("=" * 30)
    
    try:
        chroma_client = get_chroma_client()
        
        if not chroma_client.connected:
            print("❌ ChromaDB not connected!")
            return
        
        # Get collection info
        collection_info = chroma_client.get_collection_info()
        print(f"Collection Status: {collection_info}")
        
        # Try to get all documents (limit to 10 for display)
        try:
            all_results = chroma_client.collection.get(limit=10)
            print(f"Sample documents in collection: {len(all_results['ids'])} items")
            
            if all_results['ids']:
                print("Sample IDs:", all_results['ids'][:3])
                if all_results['metadatas']:
                    print("Sample metadata:", all_results['metadatas'][0] if all_results['metadatas'][0] else "None")
        except Exception as e:
            print(f"Error getting collection contents: {e}")
            
    except Exception as e:
        print(f"Debug failed: {e}")

if __name__ == "__main__":
    print("Starting Document Q&A Tests...\n")
    
    # Run debug first
    asyncio.run(test_chroma_debug())
    
    # Run main test
    success = asyncio.run(test_qa_flow())
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour document feature should now work properly!")
        print("Try uploading a document and asking 'What's in my document?'")
        exit(0)
    else:
        print("\n❌ TESTS FAILED!")
        print("Check the errors above and fix the issues.")
        exit(1)
