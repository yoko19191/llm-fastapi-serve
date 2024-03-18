import uvicorn


if __name__ == '__main__':
    print("Starting API Server...")
    uvicorn.run("api_server.main:app", host="0.0.0.0", port=8000, reload=True)