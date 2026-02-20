import typing as t
from pydantic import PrivateAttr, Field
from ...types import VersionedValue
from ..core import BaseSource
from .core import DatabaseSettings


class PostgresSource(BaseSource):
    type: t.Literal['postgres'] = "postgres"
    db: DatabaseSettings
    
    # Private cache for row data
    _row_cache: t.Dict[str, t.Dict[str, t.Any]] = PrivateAttr(default_factory=dict)
    _current_version: t.Optional[t.Any] = PrivateAttr(default=None)
    
    # Don't use the default cache system since we're implementing our own
    cache_kwargs: t.Optional[t.Dict[str, t.Any]] = None

    def get_version(
        self, 
        curr_version: t.Any, 
        requested_version: t.Any = None, 
        **kwargs
    ) -> bool:
        # If we have a specific version requested, check if it's different from current
        if requested_version is not None:
            return requested_version != curr_version
        
        # If no version requested, we need to check if there's a newer version available
        if self.db.version_table is None:
            # No versioning, always use cached data if available
            return len(self._row_cache) == 0
        
        # With versioning, we need to check if the latest version changed
        latest_version = self._get_latest_version()
        return latest_version != curr_version

    def _get_connection(self):
        """Get psycopg3 connection"""
        import psycopg
        
        conn_params = self.db.get_connection_params()
        return psycopg.connect(**conn_params)

    def _get_latest_version(self) -> t.Any:
        """Get the latest version active in the current environment"""
        if self.db.version_table is None:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Query to get the latest version for the current environment
                query = f"""
                    SELECT version 
                    FROM {self.db.version_table} 
                    WHERE environment = %s 
                    ORDER BY version DESC 
                    LIMIT 1
                """
                cur.execute(query, (self.db.environment,))
                result = cur.fetchone()
                return result[0] if result else None

    def _load_parameter_row(self, version: t.Any = None) -> t.Dict[str, t.Any]:
        """Load the entire parameter row for the given version"""
        cache_key = str(version) if version is not None else "latest"
        
        # Return cached data if available
        if cache_key in self._row_cache:
            return self._row_cache[cache_key]
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if version is not None:
                    # Load specific version
                    if self.db.version_table is not None:
                        query = f"SELECT * FROM {self.db.parameter_table} WHERE version = %s"
                        cur.execute(query, (version,))
                    else:
                        # No version table, just load the single row (assuming single row table)
                        query = f"SELECT * FROM {self.db.parameter_table}"
                        cur.execute(query)
                else:
                    # Load latest version
                    if self.db.version_table is not None:
                        latest_version = self._get_latest_version()
                        if latest_version is None:
                            return {}
                        query = f"SELECT * FROM {self.db.parameter_table} WHERE version = %s"
                        cur.execute(query, (latest_version,))
                    else:
                        # No version table, just load the single row
                        query = f"SELECT * FROM {self.db.parameter_table}"
                        cur.execute(query)
                
                # Get column names
                columns = [desc[0] for desc in cur.description]
                row = cur.fetchone()
                
                if row:
                    row_dict = dict(zip(columns, row))
                    # Cache the result
                    self._row_cache[cache_key] = row_dict
                    return row_dict
                else:
                    return {}

    def get(
        self, 
        key: str, 
        requested_version: t.Any = None, 
        **kwargs
    ) -> VersionedValue:
        # Load the parameter row for the requested or latest version
        row_data = self._load_parameter_row(requested_version)
        
        # Determine the actual version
        if requested_version is not None:
            actual_version = requested_version
        elif self.db.version_table is not None:
            # Use the version column from the row if available
            actual_version = row_data.get('version')
        else:
            actual_version = None
        
        # Get the parameter value
        if key in row_data and key != 'version':  # Don't return version column as parameter
            value = row_data[key]
        else:
            raise KeyError(f"Parameter '{key}' not found in postgres source")
        
        return VersionedValue(version=actual_version, value=value)