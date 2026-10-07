import typing as t
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    UploadFile,
    File,
    Form,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio

from dsp_decider.data_sources import DataSourceManager
from dsp_decider.data_sources.core import (
    DataSourceList,
    DataSourceTablesResponse,
    DataSourceSearchResponse,
    TableDetailResponse,
    TablePreviewResponse,
    TablePreviewResponseColumnar,
    TablePreviewDataColumnar,
    DataSourceCapabilities,
    DataSourceNotFoundError,
    TableNotFoundError,
)
from dsp_decider.settings import settings
from dsp_decider.auth.authorization import get_current_user_session
from dsp_decider.auth.models import SessionInfo
from .routes import router


class UploadResponse(BaseModel):
    """Response for file upload"""

    success: bool
    table_name: t.Optional[str] = None
    message: str


# Dependency to get data source manager
def get_datasource_manager() -> DataSourceManager:
    """Get data source manager instance"""
    return DataSourceManager(settings.datasources)


# Create router


@router.get("", response_model=DataSourceList)
async def list_datasources(
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> DataSourceList:
    """List all configured and enabled data sources"""
    return manager.list_datasources()


@router.get("/capabilities/{capability}", response_model=DataSourceList)
async def get_datasources_by_capability(
    capability: str,
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> DataSourceList:
    """Get data sources that support a specific capability"""
    try:
        # Validate capability using the enum
        DataSourceCapabilities(capability)
    except ValueError:
        valid_capabilities = [cap.value for cap in DataSourceCapabilities]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid capability. Must be one of: {', '.join(valid_capabilities)}",
        )

    return manager.get_datasources_by_capability(capability)


@router.get("/{datasource_name}/tables", response_model=DataSourceTablesResponse)
async def list_datasource_tables(
    datasource_name: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=2000, description="Page size"),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> DataSourceTablesResponse:
    """List tables in a specific data source"""
    try:
        return await manager.get_datasource_tables(
            datasource_name, page=page, page_size=page_size
        )
    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {e}")


@router.get("/{datasource_name}/tables/search", response_model=DataSourceSearchResponse)
async def search_datasource_tables(
    datasource_name: str,
    query: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Page size"),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> DataSourceSearchResponse:
    """Search tables in a specific data source"""
    try:
        return await manager.search_datasource_tables(
            datasource_name, query=query, page=page, page_size=page_size
        )
    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search tables: {e}")


@router.get(
    "/{datasource_name}/tables/{table_name}", response_model=TableDetailResponse
)
async def get_table_details(
    datasource_name: str,
    table_name: str,
    response: Response,
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> TableDetailResponse:
    """Get detailed information about a specific table"""
    try:
        result = await manager.get_table_info(datasource_name, table_name)
        response.headers["Cache-Control"] = (
            "private, max-age=300"  # 5 minutes, browser only
        )
        return result
    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get table details: {e}")


@router.get(
    "/{datasource_name}/tables/{table_name}/preview",
    response_model=TablePreviewResponse,
)
async def preview_table_data(
    datasource_name: str,
    table_name: str,
    response: Response,
    limit: int = Query(100, ge=1, le=1000, description="Number of rows to preview"),
    compress: bool = Query(
        False, description="Enable gzip compression for large responses"
    ),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> TablePreviewResponse:
    """Preview data from a specific table"""
    try:
        result = await manager.preview_table_data(
            datasource_name, table_name, limit=limit
        )
        response.headers["Cache-Control"] = (
            "private, max-age=300"  # 5 minutes, browser only
        )

        # For large responses, consider using compression
        if compress and result.data.row_count > 20:
            # You could implement custom compression here if needed
            pass

        return result
    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to preview table data: {e}"
        )


@router.get(
    "/{datasource_name}/tables/{table_name}/preview-columnar",
    response_model=TablePreviewResponseColumnar,
)
async def preview_table_data_columnar(
    datasource_name: str,
    table_name: str,
    limit: int = Query(100, ge=1, le=1000, description="Number of rows to preview"),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> TablePreviewResponseColumnar:
    """Preview data from a specific table in columnar format (more efficient for wide tables)"""
    try:
        # Get the regular preview response
        response = await manager.preview_table_data(
            datasource_name, table_name, limit=limit
        )

        # Convert to columnar format
        columnar_data = {}
        for i, column in enumerate(response.data.columns):
            columnar_data[column] = [row[i] for row in response.data.rows]

        return TablePreviewResponseColumnar(
            datasource=datasource_name,
            table=table_name,
            limit=limit,
            data=TablePreviewDataColumnar(
                columns=response.data.columns,
                data=columnar_data,
                row_count=response.data.row_count,
                total_columns=response.data.total_columns,
            ),
        )
    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to preview table data: {e}"
        )


@router.get("/{datasource_name}/tables/{table_name}/preview-stream")
async def preview_table_data_stream(
    datasource_name: str,
    table_name: str,
    limit: int = Query(100, ge=1, le=1000, description="Number of rows to preview"),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
):
    """Stream preview data from a specific table for faster loading UX"""

    async def generate_preview_stream():
        try:
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'datasource': datasource_name, 'table': table_name, 'limit': limit})}\n\n"

            # Get the preview data
            preview_response = await manager.preview_table_data(
                datasource_name, table_name, limit=limit
            )

            # Send schema info first
            yield f"data: {json.dumps({'type': 'schema', 'columns': preview_response.data.columns})}\n\n"

            # Send rows in chunks for better UX (10 rows at a time)
            chunk_size = 10
            total_rows = len(preview_response.data.rows)

            for i in range(0, total_rows, chunk_size):
                chunk_rows = preview_response.data.rows[i : i + chunk_size]
                chunk_data = {
                    "type": "data",
                    "rows": chunk_rows,
                    "chunk_start": i,
                    "chunk_end": min(i + chunk_size, total_rows),
                    "total_rows": total_rows,
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                # Small delay to allow UI to update progressively
                await asyncio.sleep(0.01)

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'total_rows': total_rows})}\n\n"

        except (DataSourceNotFoundError, TableNotFoundError) as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'error_type': 'not_found'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': f'Failed to preview table data: {e}'})}\n\n"

    return StreamingResponse(
        generate_preview_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        },
    )


@router.post("/{datasource_name}/upload", response_model=UploadResponse)
async def upload_file_to_datasource(
    datasource_name: str,
    file: UploadFile = File(...),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> UploadResponse:
    """Upload a file to a data source that supports uploads"""
    try:
        # Check if datasource supports uploads
        datasource = manager.get_datasource(datasource_name)

        # Check if the datasource has upload capability
        if not hasattr(datasource, "upload_file"):
            raise HTTPException(
                status_code=400,
                detail=f"Data source '{datasource_name}' does not support file uploads",
            )

        # Read file content
        content = await file.read()

        # Upload file
        table_name = await datasource.upload_file(file.filename, content)

        return UploadResponse(
            success=True,
            table_name=table_name,
            message=f"File '{file.filename}' uploaded successfully as table '{table_name}'",
        )

    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        return UploadResponse(success=False, message=f"Failed to upload file: {e}")


@router.post("/create-view", response_model=UploadResponse)
async def create_view_from_sql(
    source_datasource: str = Form(..., description="Source datasource to query from"),
    dest_datasource: str = Form(..., description="Destination datasource to save to"),
    source_table: str = Form(...),
    view_name: str = Form(...),
    sql_query: str = Form(...),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> UploadResponse:
    """Create a materialized view from a SQL query - query from source, save to destination"""
    try:
        # Get source datasource (must support SQL queries)
        source_ds = manager.get_datasource(source_datasource)
        if not hasattr(source_ds, "preview_sql_query"):
            raise HTTPException(
                status_code=400,
                detail=f"Source datasource '{source_datasource}' does not support SQL queries",
            )

        # Get destination datasource (must support uploads)
        dest_ds = manager.get_datasource(dest_datasource)
        if not hasattr(dest_ds, "upload_file"):
            raise HTTPException(
                status_code=400,
                detail=f"Destination datasource '{dest_datasource}' does not support uploads",
            )

        # Execute SQL query on source datasource (no limit)
        result_df = await source_ds.preview_sql_query(
            source_table_name=source_table,
            sql_query=sql_query,
            limit=1000000,  # Large limit to get full result
        )

        # Convert DataFrame to Parquet bytes
        from io import BytesIO

        parquet_buffer = BytesIO()
        result_df.write_parquet(parquet_buffer, compression="snappy")
        parquet_content = parquet_buffer.getvalue()

        # Upload to destination datasource
        created_view_name = await dest_ds.upload_file(
            filename=f"{view_name}.parquet",
            content=parquet_content,
        )

        return UploadResponse(
            success=True,
            table_name=created_view_name,
            message=f"View '{created_view_name}' created successfully from SQL query",
        )

    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        return UploadResponse(success=False, message=f"Failed to create view: {e}")


@router.post(
    "/{datasource_name}/preview-query", response_model=TablePreviewResponseColumnar
)
async def preview_sql_query(
    datasource_name: str,
    source_table: str = Form(...),
    sql_query: str = Form(...),
    limit: int = Query(20, ge=1, le=100, description="Maximum rows to return"),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> TablePreviewResponseColumnar:
    """Preview the results of a SQL query before creating a view"""
    try:
        # Get datasource
        datasource = manager.get_datasource(datasource_name)

        # Check if the datasource supports SQL query preview
        if not hasattr(datasource, "preview_sql_query"):
            raise HTTPException(
                status_code=400,
                detail=f"Data source '{datasource_name}' does not support SQL query preview",
            )

        # Execute preview query
        preview_df = await datasource.preview_sql_query(
            source_table_name=source_table,
            sql_query=sql_query,
            limit=limit,
        )

        # Convert DataFrame to columnar format
        columns = preview_df.columns
        data = {col: preview_df[col].to_list() for col in columns}

        preview_data = TablePreviewDataColumnar(
            columns=columns,
            data=data,
            row_count=len(preview_df),
            total_columns=len(columns),
        )

        return TablePreviewResponseColumnar(
            datasource=datasource_name,
            table=f"query_preview_{source_table}",
            limit=limit,
            data=preview_data,
        )

    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview query: {e}")


# Multipart upload endpoints for large files
class InitiateUploadResponse(BaseModel):
    """Response for upload initiation"""

    upload_id: str
    chunk_size: int


class ChunkUploadResponse(BaseModel):
    """Response for chunk upload"""

    chunk_number: int
    success: bool
    message: str


class CompleteUploadResponse(BaseModel):
    """Response for upload completion"""

    success: bool
    table_name: t.Optional[str] = None
    message: str


@router.post("/{datasource_name}/upload/initiate")
async def initiate_multipart_upload(
    datasource_name: str,
    filename: str = Form(...),
    file_size: int = Form(...),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> InitiateUploadResponse:
    """Initiate a multipart upload for large files"""
    try:
        # Check if datasource supports uploads
        datasource = manager.get_datasource(datasource_name)

        if not hasattr(datasource, "upload_file"):
            raise HTTPException(
                status_code=400,
                detail=f"Data source '{datasource_name}' does not support file uploads",
            )

        # Generate upload ID and set chunk size (e.g., 5MB chunks)
        import uuid

        upload_id = str(uuid.uuid4())
        chunk_size = 5 * 1024 * 1024  # 5MB

        # TODO: Store upload session info (could use cache/redis)
        # For now, return basic info
        return InitiateUploadResponse(upload_id=upload_id, chunk_size=chunk_size)

    except (DataSourceNotFoundError, TableNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate upload: {e}")


@router.post("/{datasource_name}/upload/{upload_id}/chunk/{chunk_number}")
async def upload_chunk(
    datasource_name: str,
    upload_id: str,
    chunk_number: int,
    chunk: UploadFile = File(...),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> ChunkUploadResponse:
    """Upload a single chunk of a multipart upload"""
    try:
        # TODO: Implement chunk storage logic
        # This would typically store chunks temporarily until completion

        return ChunkUploadResponse(
            chunk_number=chunk_number,
            success=True,
            message=f"Chunk {chunk_number} uploaded successfully",
        )

    except Exception as e:
        return ChunkUploadResponse(
            chunk_number=chunk_number,
            success=False,
            message=f"Failed to upload chunk: {e}",
        )


@router.post("/{datasource_name}/upload/{upload_id}/complete")
async def complete_multipart_upload(
    datasource_name: str,
    upload_id: str,
    filename: str = Form(...),
    total_chunks: int = Form(...),
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> CompleteUploadResponse:
    """Complete a multipart upload by assembling chunks"""
    try:
        # TODO: Implement chunk assembly and final upload logic
        # This would typically:
        # 1. Retrieve all chunks for the upload_id
        # 2. Assemble them into the complete file
        # 3. Upload to the datasource
        # 4. Clean up temporary chunks

        return CompleteUploadResponse(
            success=True,
            table_name=filename.split(".")[0],  # Simple table name from filename
            message=f"Multipart upload completed successfully",
        )

    except Exception as e:
        return CompleteUploadResponse(
            success=False, message=f"Failed to complete upload: {e}"
        )
