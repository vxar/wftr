"""
Questrade OAuth Authentication Handler
Handles OAuth 2.0 flow for Questrade API integration
"""
import http.server
import socketserver
import urllib.parse
import requests
import json
import threading
import time
from typing import Optional, Dict
import webbrowser
from pathlib import Path


class QuestradeOAuthHandler:
    """Handles Questrade OAuth 2.0 authentication flow"""
    
    # Questrade OAuth endpoints
    AUTHORIZATION_URL = "https://login.questrade.com/oauth2/authorize"
    TOKEN_URL = "https://login.questrade.com/oauth2/token"
    
    def __init__(self, client_id: str, redirect_uri: str = "http://localhost:8080/callback"):
        """
        Initialize OAuth handler
        
        Args:
            client_id: Your Questrade app client ID
            redirect_uri: Callback URL (must match what you registered in Questrade)
        """
        # Validate and clean client ID
        if not client_id:
            raise ValueError("Client ID cannot be empty")
        
        # Remove any whitespace
        client_id = client_id.strip()
        
        # Questrade client IDs are typically alphanumeric strings
        # Validate basic format (should be non-empty alphanumeric)
        if not client_id or len(client_id) < 10:
            raise ValueError(f"Client ID appears invalid. Questrade client IDs are typically longer alphanumeric strings. Got: '{client_id}'")
        
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.authorization_code = None
        self.tokens = None
        self.server = None
        self.server_thread = None
        self.callback_received = False
        self.error_message = None
        
        print(f"[INFO] Using Client ID: {client_id[:10]}...{client_id[-5:] if len(client_id) > 15 else ''}")
        print(f"[INFO] Callback URL: {redirect_uri}")
    
    def get_authorization_url(self) -> str:
        """
        Generate the authorization URL for user to visit
        
        Returns:
            URL string for user to authorize the application
        """
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri
        }
        auth_url = f"{self.AUTHORIZATION_URL}?{urllib.parse.urlencode(params)}"
        
        # Debug: Show the URL being generated (without exposing full client ID)
        print(f"[DEBUG] Authorization URL generated:")
        print(f"  Base URL: {self.AUTHORIZATION_URL}")
        print(f"  Client ID length: {len(self.client_id)} characters")
        print(f"  Redirect URI: {self.redirect_uri}")
        print(f"  Full URL (first 100 chars): {auth_url[:100]}...")
        
        return auth_url
    
    def start_callback_server(self, port: int = 8080):
        """
        Start a local HTTP server to receive the OAuth callback
        
        Args:
            port: Port number for the callback server (default: 8080)
        """
        class CallbackHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, oauth_handler=None, **kwargs):
                self.oauth_handler = oauth_handler
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                """Handle GET request from OAuth callback"""
                # Get the full request line to see everything
                full_request = self.requestline if hasattr(self, 'requestline') else self.path
                
                parsed_path = urllib.parse.urlparse(self.path)
                
                # Handle both /callback and /callback/ (with trailing slash)
                path_normalized = parsed_path.path.rstrip('/')
                
                # Debug: Print what we received
                print(f"\n[DEBUG] Received callback:")
                print(f"  Full request line: {full_request}")
                print(f"  Full path: {self.path}")
                print(f"  Parsed path: {parsed_path.path}")
                print(f"  Query string: {parsed_path.query}")
                print(f"  Fragment: {parsed_path.fragment}")
                print(f"  Normalized path: {path_normalized}")
                
                if path_normalized == '/callback':
                    # Parse query parameters (from ?param=value)
                    query_params = urllib.parse.parse_qs(parsed_path.query)
                    
                    # Also check fragment (from #param=value) - some OAuth providers use this
                    if parsed_path.fragment:
                        fragment_params = urllib.parse.parse_qs(parsed_path.fragment)
                        # Merge fragment params into query params
                        for key, value in fragment_params.items():
                            if key not in query_params:
                                query_params[key] = value
                    
                    print(f"  Query params: {query_params}")
                    
                    # Check for authorization code
                    if 'code' in query_params:
                        code = query_params['code'][0]
                        print(f"  [SUCCESS] Authorization code received: {code[:20]}...")
                        self.oauth_handler.authorization_code = code
                        self.oauth_handler.callback_received = True
                        
                        # Send success response
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b"""
                        <html>
                        <head><title>Authorization Successful</title></head>
                        <body>
                        <h1>Authorization Successful!</h1>
                        <p>You can close this window and return to your application.</p>
                        <script>setTimeout(function(){window.close();}, 3000);</script>
                        </body>
                        </html>
                        """)
                    elif 'error' in query_params:
                        # Handle error
                        error = query_params['error'][0]
                        error_description = query_params.get('error_description', [''])[0]
                        print(f"  [ERROR] Authorization error: {error} - {error_description}")
                        self.oauth_handler.error_message = f"{error}: {error_description}"
                        self.oauth_handler.callback_received = True
                        
                        # Send error response
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(f"""
                        <html>
                        <head><title>Authorization Failed</title></head>
                        <body>
                        <h1>Authorization Failed</h1>
                        <p>Error: {error}</p>
                        <p>{error_description}</p>
                        </body>
                        </html>
                        """.encode())
                    else:
                        # Invalid callback - no code or error
                        print(f"  [WARNING] No 'code' or 'error' parameter found in callback")
                        print(f"  Available parameters: {list(query_params.keys())}")
                        
                        # Send error response with debug info
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        debug_info = f"""
                        <p><strong>Debug Information:</strong></p>
                        <ul>
                            <li>Full path: {self.path}</li>
                            <li>Query string: {parsed_path.query}</li>
                            <li>Query params: {query_params}</li>
                        </ul>
                        """
                        self.wfile.write(f"""
                        <html>
                        <head><title>Invalid Callback</title></head>
                        <body>
                        <h1>Invalid Callback</h1>
                        <p>No authorization code or error received.</p>
                        {debug_info}
                        <p><strong>Possible issues:</strong></p>
                        <ul>
                            <li>Callback URL mismatch - check Questrade app registration</li>
                            <li>User cancelled authorization</li>
                            <li>Questrade redirect format changed</li>
                        </ul>
                        </body>
                        </html>
                        """.encode())
                else:
                    # Not a callback path - might be a browser request for root or favicon
                    print(f"  [INFO] Non-callback path requested: {parsed_path.path}")
                    if parsed_path.path == '/' or parsed_path.path == '/favicon.ico':
                        # Send a simple response for root or favicon
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b"""
                        <html>
                        <head><title>Questrade OAuth</title></head>
                        <body>
                        <h1>Questrade OAuth Callback Server</h1>
                        <p>Waiting for OAuth callback...</p>
                        </body>
                        </html>
                        """)
                    else:
                        # Default 404
                        self.send_response(404)
                        self.end_headers()
            
            def log_message(self, format, *args):
                """Suppress server logs"""
                pass
        
        # Create handler with reference to oauth_handler
        handler = lambda *args, **kwargs: CallbackHandler(*args, oauth_handler=self, **kwargs)
        
        try:
            self.server = socketserver.TCPServer(("", port), handler)
            self.server.allow_reuse_address = True
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            print(f"Callback server started on http://localhost:{port}/callback")
        except OSError as e:
            if "Address already in use" in str(e):
                raise ValueError(f"Port {port} is already in use. Please choose a different port or close the application using it.")
            raise
    
    def stop_callback_server(self):
        """Stop the callback server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("Callback server stopped")
    
    def exchange_code_for_tokens(self) -> Dict:
        """
        Exchange authorization code for access tokens
        
        Returns:
            Dictionary containing access_token, refresh_token, token_type, expires_in, etc.
        """
        if not self.authorization_code:
            raise ValueError("No authorization code received. Please complete the authorization flow first.")
        
        # Questrade uses a GET request for token exchange (unlike standard OAuth2)
        params = {
            'client_id': self.client_id,
            'code': self.authorization_code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.get(self.TOKEN_URL, params=params)
        
        if response.status_code == 200:
            self.tokens = response.json()
            return self.tokens
        else:
            error_msg = f"Token exchange failed: {response.status_code} - {response.text}"
            raise Exception(error_msg)
    
    def refresh_access_token(self, refresh_token: str) -> Dict:
        """
        Refresh the access token using refresh token
        
        Args:
            refresh_token: The refresh token from previous authentication
            
        Returns:
            Dictionary containing new access_token, refresh_token, etc.
        """
        params = {
            'client_id': self.client_id,
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        }
        
        response = requests.get(self.TOKEN_URL, params=params)
        
        if response.status_code == 200:
            self.tokens = response.json()
            return self.tokens
        else:
            error_msg = f"Token refresh failed: {response.status_code} - {response.text}"
            raise Exception(error_msg)
    
    def authenticate(self, auto_open_browser: bool = True, timeout: int = 300) -> Dict:
        """
        Complete the full OAuth authentication flow
        
        Args:
            auto_open_browser: Automatically open browser to authorization URL
            timeout: Maximum time to wait for callback (seconds)
            
        Returns:
            Dictionary containing tokens
        """
        # Start callback server
        port = int(urllib.parse.urlparse(self.redirect_uri).port or 8080)
        self.start_callback_server(port)
        
        try:
            # Get authorization URL
            auth_url = self.get_authorization_url()
            print(f"\n{'='*80}")
            print("Questrade OAuth Authentication")
            print(f"{'='*80}")
            print(f"Please visit this URL to authorize the application:")
            print(f"\n{auth_url}\n")
            print(f"{'='*80}\n")
            
            # Open browser if requested
            if auto_open_browser:
                print("Opening browser...")
                webbrowser.open(auth_url)
            
            # Wait for callback
            print("Waiting for authorization callback...")
            start_time = time.time()
            while not self.callback_received:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Authorization timeout after {timeout} seconds")
                time.sleep(0.5)
            
            # Check for errors
            if self.error_message:
                raise Exception(f"Authorization error: {self.error_message}")
            
            if not self.authorization_code:
                raise Exception("No authorization code received")
            
            print("Authorization code received. Exchanging for tokens...")
            
            # Exchange code for tokens
            tokens = self.exchange_code_for_tokens()
            
            print("Authentication successful!")
            print(f"Access token expires in: {tokens.get('expires_in', 'N/A')} seconds")
            print(f"API server: {tokens.get('api_server', 'N/A')}")
            
            return tokens
            
        finally:
            # Stop callback server
            self.stop_callback_server()
    
    def save_tokens(self, filepath: str = "questrade_tokens.json"):
        """
        Save tokens to a JSON file
        
        Args:
            filepath: Path to save tokens file
        """
        if not self.tokens:
            raise ValueError("No tokens to save. Please authenticate first.")
        
        # Don't save the full token data in a way that could be committed
        # In production, use environment variables or secure storage
        token_data = {
            'access_token': self.tokens.get('access_token'),
            'refresh_token': self.tokens.get('refresh_token'),
            'token_type': self.tokens.get('token_type'),
            'expires_in': self.tokens.get('expires_in'),
            'api_server': self.tokens.get('api_server'),
            'expires_at': time.time() + self.tokens.get('expires_in', 0) if self.tokens.get('expires_in') else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(token_data, f, indent=2)
        
        print(f"Tokens saved to {filepath}")
        print("WARNING: Keep this file secure and do not commit it to version control!")
    
    def load_tokens(self, filepath: str = "questrade_tokens.json") -> Dict:
        """
        Load tokens from a JSON file
        
        Args:
            filepath: Path to tokens file
            
        Returns:
            Dictionary containing tokens
        """
        with open(filepath, 'r') as f:
            token_data = json.load(f)
        
        self.tokens = token_data
        return token_data


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python questrade_oauth.py <client_id> [redirect_uri]")
        print("\nExample:")
        print("  python questrade_oauth.py your_client_id_here")
        print("  python questrade_oauth.py your_client_id_here http://localhost:8080/callback")
        sys.exit(1)
    
    client_id = sys.argv[1]
    redirect_uri = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080/callback"
    
    # Create OAuth handler
    oauth = QuestradeOAuthHandler(client_id=client_id, redirect_uri=redirect_uri)
    
    try:
        # Authenticate
        tokens = oauth.authenticate()
        
        # Save tokens
        oauth.save_tokens()
        
        print("\n" + "="*80)
        print("Authentication Complete!")
        print("="*80)
        print(f"API Server: {tokens.get('api_server')}")
        print(f"Token Type: {tokens.get('token_type')}")
        print(f"Expires In: {tokens.get('expires_in')} seconds")
        print("\nYou can now use these tokens to make API calls to Questrade.")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nAuthentication cancelled by user.")
        oauth.stop_callback_server()
    except Exception as e:
        print(f"\n\nError: {e}")
        oauth.stop_callback_server()
        sys.exit(1)


if __name__ == "__main__":
    main()

