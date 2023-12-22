if __name__ == "__main__":
    import sys
    proc = sys.argv[1]
    if proc == "train":
        from engine.server import train
        train.train()
        sys.exit(0)
    elif proc == "serve":
        from engine.server import serve
        serve.start_server()
    else:
        raise ValueError(f"Engine server must be started with either train or serve but found {proc}")
