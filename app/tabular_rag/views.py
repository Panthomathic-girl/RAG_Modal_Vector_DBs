from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from app.tabular_rag.utils import _enforce_size_limit, _parse_csv, get_postgres_connection, get_mongo_connection
from app.tabular_rag.rag import upsert_texts, query_vector_store
from app.config import Settings
import google.generativeai as genai
import json
import psycopg2.extras
import re
import logging
import pandas as pd  # Added for NL2Pandas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=Settings.GOOGLE_API_KEY)

router = APIRouter(prefix="/tabular_rag", tags=["Tabular RAG"])

# Global in-memory storage for CSV data (for NL2Pandas)
CSV_DATA = None

def clean_sql_query(sql_text: str) -> str:
    """Remove markdown code block markers and other non-SQL content from generated SQL."""
    cleaned = re.sub(r'```(?:sql)?\s*', '', sql_text, flags=re.MULTILINE)
    cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE)
    cleaned = ' '.join(cleaned.strip().split())
    if not cleaned.upper().startswith('SELECT'):
        raise ValueError("Generated query is not a valid SELECT query.")
    return cleaned

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(..., description="Upload a .csv file")):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only .csv files are supported.")
    try:
        content = _enforce_size_limit(file)
        data = _parse_csv(content)
        global CSV_DATA
        CSV_DATA = pd.DataFrame(data)  # Store as pandas DataFrame for NL2Pandas
        stored_count = upsert_texts(data, file.filename, "csv")
        return JSONResponse({
            "filename": file.filename,
            "total_rows": len(data),
            "stored_vectors": stored_count,
            "status": "success"
        })
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/upload-sql")
async def upload_sql(query: str = Query(..., description="SQL query to fetch data")):
    """API 2: Upload data from PostgreSQL using a query."""
    conn = get_postgres_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            stored_count = upsert_texts(data, "sql_data", "sql")
            return JSONResponse({
                "query": query,
                "total_rows": len(data),
                "stored_vectors": stored_count,
                "status": "success"
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL query failed: {str(e)}")
    finally:
        conn.close()

@router.post("/upload-nosql")
async def upload_nosql(collection: str = Query(..., description="MongoDB collection name")):
    """API 3: Upload data from MongoDB collection."""
    db = get_mongo_connection()
    try:
        data = list(db[collection].find())
        stored_count = upsert_texts(data, "nosql_data", "nosql")
        return JSONResponse({
            "collection": collection,
            "total_documents": len(data),
            "stored_vectors": stored_count,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB query failed: {str(e)}")

@router.get("/query")
async def query_tabular(
    query: str = Query(..., description="Query text"), 
    source: str = Query("csv", enum=["csv", "sql", "nosql"]), 
    top_k: int = Query(5, ge=1, le=20),
    method: str = Query("rag", enum=["rag", "nl2sql", "nl2pandas", "nl2mongo"], description="Method for querying: rag (vector search), nl2sql (for sql), nl2pandas (for csv), or nl2mongo (for nosql)")
):
    async def event_stream():
        try:
            if source == "csv" and method == "nl2pandas":
                # NL2Pandas for CSV: Generate pandas code, execute on in-memory DataFrame
                if CSV_DATA is None:
                    yield f"data: {json.dumps({'error': 'No CSV data uploaded yet.'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                
                schema = "DataFrame with columns: id (string), name (string), age (string), city (string)"
                
                pandas_gen_prompt = f"""
                You are a pandas expert. Generate valid Python pandas code to answer the user's question using the DataFrame 'df'.
                Use df = CSV_DATA.copy() at the start.
                Treat all columns as strings; convert 'age' to numeric (pd.to_numeric) for calculations.
                For aggregations, use meaningful variables, e.g., average_age = df[df['city'] == 'New York']['age'].astype(float).mean()
                For filters, use df.query or boolean masking, e.g., df[df['city'] == 'Boston']
                For 'who' questions, select 'name' and relevant columns like 'age', 'city'.
                For oldest/youngest, use df.nlargest/nlargest or sort_values on 'age' after converting to numeric.
                Set a 'result' variable with the final output (DataFrame, scalar, or list).
                Do not use external libraries except pandas (pd).
                Do not fabricate data; use only the DataFrame.
                Output ONLY the Python code, nothing else.
                Schema:
                {schema}
                
                User question: {query}
                
                Code:
                """
                
                model = genai.GenerativeModel('gemini-2.5-flash')
                pandas_response = model.generate_content(pandas_gen_prompt)
                generated_code = pandas_response.text.strip()
                logger.info(f"Generated Pandas code: {generated_code}")
                
                # Execute the generated code
                try:
                    local_vars = {"CSV_DATA": CSV_DATA, "pd": pd}
                    exec(generated_code, {}, local_vars)
                    result = local_vars.get('result', None)
                    if result is None:
                        context = "No result from code execution."
                    elif isinstance(result, pd.DataFrame):
                        context = "\n".join([", ".join(f"{k}: {v}" for k, v in row.items()) for row in result.to_dict('records')])
                    elif isinstance(result, (int, float)):
                        context = f"Result: {result}"
                    elif isinstance(result, list):
                        context = "\n".join([str(item) for item in result])
                    else:
                        context = str(result)
                except Exception as e:
                    logger.error(f"Pandas execution failed: {str(e)}")
                    yield f"data: {json.dumps({'error': f'Pandas execution failed: {str(e)}. Generated code: {generated_code}'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                
            elif source == "sql" and method == "nl2sql":
                # Unchanged SQL logic
                schema = """
                Table: large_test
                Columns:
                - id: integer (primary key)
                - name: text
                - age: integer
                - city: text
                """
                
                sql_gen_prompt = f"""
                You are a SQL expert. Generate a valid PostgreSQL SELECT query to answer the user's question exactly.
                Use the exact column names: id, name, age, city.
                For aggregations, use AS to alias results meaningfully, e.g., AVG(age) AS average_age, MAX(age) AS max_age, COUNT(*) AS count.
                For filters, include WHERE clauses with ILIKE for case-insensitive text comparisons, e.g., city ILIKE 'new york'.
                For 'who' questions, select name and other relevant columns like age, city.
                For oldest/youngest, use ORDER BY age DESC/ASC LIMIT if single, or subquery for all with max/min age.
                Do not use any DML or DDL statements. Only SELECT.
                Do not include markdown code block markers (e.g., ```sql or ```).
                Schema:
                {schema}
                
                User question: {query}
                
                Output ONLY the SQL query, nothing else.
                """
                
                model = genai.GenerativeModel('gemini-2.5-flash')
                sql_response = model.generate_content(sql_gen_prompt)
                generated_sql = clean_sql_query(sql_response.text)
                logger.info(f"Generated SQL: {generated_sql}")
                
                conn = get_postgres_connection()
                try:
                    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                        cursor.execute(generated_sql)
                        rows = cursor.fetchall()
                        data = [dict(row) for row in rows]
                except Exception as e:
                    logger.error(f"SQL execution failed: {str(e)}")
                    yield f"data: {json.dumps({'error': f'SQL execution failed: {str(e)}. Generated SQL: {generated_sql}'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                finally:
                    conn.close()
                
                context = "\n".join([", ".join(f"{k}: {v}" for k, v in row.items()) for row in data])
                if not context:
                    context = "No data found."
                
            elif source == "nosql" and method == "nl2mongo":
                # NL2Mongo for NoSQL: Generate PyMongo code, execute on MongoDB collection
                db = get_mongo_connection()
                collection_name = "large_test"  # Assume default collection; can be made dynamic
                schema = "Collection with documents having fields: id (string), name (string), age (string), city (string)"
                
                mongo_gen_prompt = f"""
                You are a MongoDB expert. Generate valid Python PyMongo code to answer the user's question using the collection 'collection'.
                Use collection = db['{collection_name}'] at the start.
                Treat all fields as strings; convert 'age' to float for calculations.
                For aggregations, use aggregation pipelines, e.g., result = list(collection.aggregate([{{'$match': {{'city': 'New York'}}}}, {{'$group': {{'_id': None, 'average_age': {{'$avg': {{'$toDouble': '$age'}}}}}} ]]))
                For filters, use find, e.g., result = list(collection.find({{'city': 'Boston'}}))
                For 'who' questions, project 'name' and relevant fields like 'age', 'city'.
                For oldest/youngest, use sort and limit or aggregation for max/min age.
                Set a 'result' variable with the final output (list of dicts, scalar, or list).
                Do not use external libraries except pymongo (implicitly).
                Do not fabricate data; use only the collection.
                Output ONLY the Python code, nothing else.
                Schema:
                {schema}
                
                User question: {query}
                
                Code:
                """
                
                model = genai.GenerativeModel('gemini-2.5-flash')
                mongo_response = model.generate_content(mongo_gen_prompt)
                generated_code = mongo_response.text.strip()
                logger.info(f"Generated Mongo code: {generated_code}")
                
                # Execute the generated code
                try:
                    local_vars = {"db": db, "collection_name": collection_name}
                    exec(generated_code, {}, local_vars)
                    result = local_vars.get('result', None)
                    if result is None:
                        context = "No result from code execution."
                    elif isinstance(result, list):
                        context = "\n".join([", ".join(f"{k}: {v}" for k, v in doc.items()) for doc in result])
                    elif isinstance(result, (int, float)):
                        context = f"Result: {result}"
                    else:
                        context = str(result)
                except Exception as e:
                    logger.error(f"Mongo execution failed: {str(e)}")
                    # Fallback to RAG if Mongo execution fails
                    logger.warning("Falling back to RAG for NoSQL query")
                    results = query_vector_store(query, top_k=top_k, filter={"source": source})
                    context_texts = [m['metadata']['text'] for m in results.get('matches', []) if 'text' in m.get('metadata', {})]
                    if not context_texts:
                        yield f"data: {json.dumps({'error': 'No relevant information found.'})}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                        return
                    context = "\n\n".join(context_texts)
                    logger.info(f"RAG fallback context: {context}")
                
            else:
                # RAG for other cases
                results = query_vector_store(query, top_k=top_k, filter={"source": source})
                
                context_texts = [m['metadata']['text'] for m in results.get('matches', []) if 'text' in m.get('metadata', {})]
                if not context_texts:
                    yield f"data: {json.dumps({'error': 'No relevant information found.'})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                
                context = "\n\n".join(context_texts)
                logger.info(f"RAG context: {context}")
            
            prompt = f"""Based on the tabular data, answer concisely in natural language like a helpful chatbot:

Context:
{context}

Query: {query}

Answer:"""
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            full_response = response.text.strip()
            
            yield f"data: {json.dumps({'chunk': full_response})}\n\n"
            yield f"data: {json.dumps({'message': full_response})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# from fastapi import APIRouter, UploadFile, File, HTTPException, Query
# from fastapi.responses import JSONResponse, StreamingResponse
# from app.tabular_rag.utils import _enforce_size_limit, _parse_csv, get_postgres_connection, get_mongo_connection
# from app.tabular_rag.rag import upsert_texts, query_vector_store
# from app.config import Settings
# import google.generativeai as genai
# import json
# import psycopg2.extras
# import re
# import logging
# import pandas as pd

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# genai.configure(api_key=Settings.GOOGLE_API_KEY)

# router = APIRouter(prefix="/tabular_rag", tags=["Tabular RAG"])

# # Global in-memory storage for CSV data (for NL2Pandas)
# CSV_DATA = None

# def clean_sql_query(sql_text: str) -> str:
#     """Remove markdown code block markers and other non-SQL content from generated SQL."""
#     cleaned = re.sub(r'```(?:sql)?\s*', '', sql_text, flags=re.MULTILINE)
#     cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE)
#     cleaned = ' '.join(cleaned.strip().split())
#     if not cleaned.upper().startswith('SELECT'):
#         raise ValueError("Generated query is not a valid SELECT query.")
#     return cleaned

# @router.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(..., description="Upload a .csv file")):
#     if not file.filename.lower().endswith(".csv"):
#         raise HTTPException(status_code=415, detail="Only .csv files are supported.")
#     try:
#         content = _enforce_size_limit(file)
#         data = _parse_csv(content)
#         global CSV_DATA
#         CSV_DATA = pd.DataFrame(data)  # Store as pandas DataFrame for NL2Pandas
#         logger.info(f"CSV uploaded: {file.filename}, rows: {len(data)}, CSV_DATA shape: {CSV_DATA.shape}")
#         stored_count = upsert_texts(data, file.filename, "csv")
#         return JSONResponse({
#             "filename": file.filename,
#             "total_rows": len(data),
#             "stored_vectors": stored_count,
#             "status": "success"
#         })
#     except HTTPException as e:
#         logger.error(f"Upload error: {str(e)}")
#         raise e
#     except Exception as e:
#         logger.error(f"Processing failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# @router.post("/upload-sql")
# async def upload_sql(query: str = Query(..., description="SQL query to fetch data")):
#     conn = get_postgres_connection()
#     try:
#         with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
#             cursor.execute(query)
#             rows = cursor.fetchall()
#             data = [dict(row) for row in rows]
#             stored_count = upsert_texts(data, "sql_data", "sql")
#             return JSONResponse({
#                 "query": query,
#                 "total_rows": len(data),
#                 "stored_vectors": stored_count,
#                 "status": "success"
#             })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"SQL query failed: {str(e)}")
#     finally:
#         conn.close()

# @router.post("/upload-nosql")
# async def upload_nosql(collection: str = Query(..., description="MongoDB collection name")):
#     db = get_mongo_connection()
#     try:
#         data = list(db[collection].find())
#         stored_count = upsert_texts(data, "nosql_data", "nosql")
#         return JSONResponse({
#             "collection": collection,
#             "total_documents": len(data),
#             "stored_vectors": stored_count,
#             "status": "success"
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"MongoDB query failed: {str(e)}")

# @router.get("/query")
# async def query_tabular(
#     query: str = Query(..., description="Query text"), 
#     source: str = Query("csv", enum=["csv", "sql", "nosql"]), 
#     top_k: int = Query(20, ge=1, le=50),
#     method: str = Query("rag", enum=["rag", "nl2sql", "nl2pandas"], description="Method for querying: rag (vector search), nl2sql (for sql), or nl2pandas (for csv)")
# ):
#     async def event_stream():
#         try:
#             if source == "csv" and method == "nl2pandas":
#                 # NL2Pandas for CSV: Generate pandas code, execute on in-memory DataFrame
#                 logger.info(f"CSV_DATA status: {'set' if CSV_DATA is not None else 'None'}")
#                 if CSV_DATA is None:
#                     logger.warning("CSV_DATA is None, falling back to RAG method")
#                     # Fallback to RAG if CSV_DATA is not available
#                     results = query_vector_store(query, top_k=top_k, filter={"source": source})
#                     context_texts = [m['metadata']['text'] for m in results.get('matches', []) if 'text' in m.get('metadata', {})]
#                     if not context_texts:
#                         yield f"data: {json.dumps({'error': 'No CSV data uploaded yet and no relevant information found in Pinecone.'})}\n\n"
#                         yield "event: done\ndata: [DONE]\n\n"
#                         return
#                     context = "\n\n".join(context_texts)
#                     logger.info(f"RAG fallback context: {context}")
#                 else:
#                     schema = "DataFrame with columns: id (string), name (string), age (string), city (string)"
                    
#                     pandas_gen_prompt = f"""
#                     You are a pandas expert. Generate valid Python pandas code to answer the user's question using the DataFrame 'df'.
#                     Use df = CSV_DATA.copy() at the start.
#                     Treat all columns as strings; convert 'age' to numeric (pd.to_numeric) for calculations.
#                     For aggregations, use meaningful variables, e.g., average_age = df[df['city'] == 'New York']['age'].astype(float).mean().
#                     For filters, use df.query or boolean masking, e.g., df[df['city'] == 'Boston'].
#                     For 'who' questions, select 'name' and relevant columns like 'age', 'city'.
#                     For oldest/youngest, use df.nlargest/nlargest or sort_values on 'age' after converting to numeric.
#                     Set a 'result' variable with the final output (DataFrame, scalar, or list).
#                     Do not use external libraries except pandas (pd).
#                     Do not fabricate data; use only the DataFrame.
#                     Output ONLY the Python code, nothing else.
#                     Schema:
#                     {schema}
                    
#                     User question: {query}
                    
#                     Code:
#                     """
                    
#                     model = genai.GenerativeModel('gemini-2.5-flash')
#                     pandas_response = model.generate_content(pandas_gen_prompt)
#                     generated_code = pandas_response.text.strip()
#                     logger.info(f"Generated Pandas code: {generated_code}")
                    
#                     # Execute the generated code
#                     try:
#                         local_vars = {"CSV_DATA": CSV_DATA, "pd": pd}
#                         exec(generated_code, {}, local_vars)
#                         result = local_vars.get('result', None)
#                         if result is None:
#                             context = "No result from code execution."
#                         elif isinstance(result, pd.DataFrame):
#                             context = "\n".join([", ".join(f"{k}: {v}" for k, v in row.items()) for row in result.to_dict('records')])
#                         elif isinstance(result, (int, float)):
#                             context = f"Result: {result}"
#                         elif isinstance(result, list):
#                             context = "\n".join([str(item) for item in result])
#                         else:
#                             context = str(result)
#                         logger.info(f"Pandas result context: {context}")
#                     except Exception as e:
#                         logger.error(f"Pandas execution failed: {str(e)}")
#                         yield f"data: {json.dumps({'error': f'Pandas execution failed: {str(e)}. Generated code: {generated_code}'})}\n\n"
#                         yield "event: done\ndata: [DONE]\n\n"
#                         return
                
#             elif source == "sql" and method == "nl2sql":
#                 # Unchanged SQL logic
#                 schema = """
#                 Table: large_test
#                 Columns:
#                 - id: integer (primary key)
#                 - name: text
#                 - age: integer
#                 - city: text
#                 """
                
#                 sql_gen_prompt = f"""
#                 You are a SQL expert. Generate a valid PostgreSQL SELECT query to answer the user's question exactly.
#                 Use the exact column names: id, name, age, city.
#                 For aggregations, use AS to alias results meaningfully, e.g., AVG(age) AS average_age, MAX(age) AS max_age, COUNT(*) AS count.
#                 For filters, include WHERE clauses with ILIKE for case-insensitive text comparisons, e.g., city ILIKE 'new york'.
#                 For 'who' questions, select name and other relevant columns like age, city.
#                 For oldest/youngest, use ORDER BY age DESC/ASC LIMIT if single, or subquery for all with max/min age.
#                 Do not use any DML or DDL statements. Only SELECT.
#                 Do not include markdown code block markers (e.g., ```sql or ```).
#                 Schema:
#                 {schema}
                
#                 User question: {query}
                
#                 Output ONLY the SQL query, nothing else.
#                 """
                
#                 model = genai.GenerativeModel('gemini-2.5-flash')
#                 sql_response = model.generate_content(sql_gen_prompt)
#                 generated_sql = clean_sql_query(sql_response.text)
#                 logger.info(f"Generated SQL: {generated_sql}")
                
#                 conn = get_postgres_connection()
#                 try:
#                     with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
#                         cursor.execute(generated_sql)
#                         rows = cursor.fetchall()
#                         data = [dict(row) for row in rows]
#                 except Exception as e:
#                     logger.error(f"SQL execution failed: {str(e)}")
#                     yield f"data: {json.dumps({'error': f'SQL execution failed: {str(e)}. Generated SQL: {generated_sql}'})}\n\n"
#                     yield "event: done\ndata: [DONE]\n\n"
#                     return
#                 finally:
#                     conn.close()
                
#                 context = "\n".join([", ".join(f"{k}: {v}" for k, v in row.items()) for row in data])
#                 if not context:
#                     context = "No data found."
                
#             else:
#                 # RAG for csv (fallback) or nosql
#                 results = query_vector_store(query, top_k=top_k, filter={"source": source})
                
#                 context_texts = [m['metadata']['text'] for m in results.get('matches', []) if 'text' in m.get('metadata', {})]
#                 if not context_texts:
#                     yield f"data: {json.dumps({'error': 'No relevant information found.'})}\n\n"
#                     yield "event: done\ndata: [DONE]\n\n"
#                     return
                
#                 context = "\n\n".join(context_texts)
#                 logger.info(f"RAG context: {context}")
            
#             prompt = f"""
#             You are a precise chatbot. Answer the user's query concisely in natural language based EXCLUSIVELY on the provided tabular data. 
#             Do NOT add any information, names, or data not explicitly present in the context. 
#             If the context contains numerical results (e.g., counts or averages), use them directly with exact values. 
#             For lists of names, include all names provided without truncation unless explicitly stated. 
#             If no data is found, state so clearly. 
#             Do not fabricate or infer additional data.
            
#             Context:
#             {context}
            
#             Query: {query}
            
#             Answer:
#             """
            
#             model = genai.GenerativeModel('gemini-2.5-flash')
#             response = model.generate_content(prompt)
#             full_response = response.text.strip()
            
#             yield f"data: {json.dumps({'chunk': full_response})}\n\n"
#             yield f"data: {json.dumps({'message': full_response})}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         except Exception as e:
#             logger.error(f"Query processing failed: {str(e)}")
#             yield f"data: {json.dumps({'error': str(e)})}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"

#     return StreamingResponse(event_stream(), media_type="text/event-stream")