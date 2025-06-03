#!/usr/bin/env python3
"""
Serveur HTTP simple pour servir les fichiers statiques sans cache.
"""
import http.server
import socketserver
import os
from urllib.parse import urlparse

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Ajouter des en-têtes pour empêcher la mise en cache
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def guess_type(self, path):
        # Forcer le type MIME pour les fichiers JavaScript
        if path.endswith('.js'):
            return 'application/javascript'
        return super().guess_type(path)

if __name__ == "__main__":
    PORT = 8080
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
        print(f"Serveur HTTP démarré sur le port {PORT}")
        print(f"Ouvrir http://localhost:{PORT}/websocket_client.html dans votre navigateur")
        httpd.serve_forever()