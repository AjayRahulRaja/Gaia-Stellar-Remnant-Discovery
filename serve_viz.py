import http.server
import socketserver
import webbrowser
import os

PORT = 8000
DIRECTORY = "viz"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

def start_server():
    # We run from the root, but index is in viz/
    # So we should be able to fetch results/viz_data.json relative to the root
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server started at http://localhost:{PORT}/viz/")
        print("Click the link above to explore the 3D Star Map.")
        webbrowser.open(f"http://localhost:{PORT}/viz/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.shutdown()

if __name__ == "__main__":
    start_server()
