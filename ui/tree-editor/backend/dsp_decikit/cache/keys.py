class CacheKeys:
    @staticmethod
    def auth_session(session_id: str) -> str:
        return f"auth.session:{session_id}"

    @staticmethod
    def oidc_login_session(session_id: str) -> str:
        return f"oidc.login.sessions:{session_id}"

    @staticmethod
    def csrf_token(token: str) -> str:
        return f"csrf:{token}"

    # Datasource cache keys
    @staticmethod
    def datasource_schema(datasource_name: str, table_name: str) -> str:
        return f"datasource.schema:{datasource_name}:{table_name}"

    @staticmethod
    def datasource_preview(datasource_name: str, table_name: str, limit: str) -> str:
        return f"datasource.preview:{datasource_name}:{table_name}"

    @staticmethod
    def datasource_table_info(datasource_name: str, table_name: str) -> str:
        return f"datasource.info:{datasource_name}:{table_name}"
