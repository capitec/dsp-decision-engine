if __name__ == "__main__":
    import sys

    proc = sys.argv[1]
    if proc == "serve":
        from spockflow.inference.server import serve

        serve.start_server()
    else:
        raise ValueError(f"SpockFlow can only be run to serve models. Found {proc}")
