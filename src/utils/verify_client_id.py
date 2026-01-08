"""
Helper script to verify Questrade Client ID format
"""
import sys


def verify_client_id(client_id: str):
    """Verify if a client ID looks valid"""
    if not client_id:
        print("ERROR: Client ID is empty")
        return False
    
    # Remove whitespace
    client_id = client_id.strip()
    
    print(f"\n{'='*60}")
    print("Questrade Client ID Verification")
    print(f"{'='*60}")
    print(f"Client ID: {client_id}")
    print(f"Length: {len(client_id)} characters")
    
    # Check for common issues
    issues = []
    
    if len(client_id) < 10:
        issues.append(f"WARNING: Client ID seems too short (typically 20+ characters)")
    
    if ' ' in client_id:
        issues.append("ERROR: Client ID contains spaces - remove any spaces")
    
    if '\n' in client_id or '\r' in client_id:
        issues.append("ERROR: Client ID contains line breaks")
    
    if client_id.startswith('http') or client_id.startswith('www'):
        issues.append("ERROR: Client ID should not be a URL")
    
    # Check if it looks like it might be a token instead of client ID
    if len(client_id) > 100:
        issues.append("WARNING: Client ID seems very long - might be a token instead")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\nClient ID format looks valid")
    
    print(f"\n{'='*60}")
    print("How to get your Client ID:")
    print("1. Log in to Questrade account")
    print("2. Go to API Centre (dropdown under your name)")
    print("3. Click 'Register a personal app' or view existing apps")
    print("4. Copy the 'Client ID' or 'Application ID'")
    print("5. It should be a long alphanumeric string (no spaces)")
    print(f"{'='*60}\n")
    
    return len(issues) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_client_id.py <client_id>")
        print("\nExample:")
        print("  python verify_client_id.py rXK1khHLQ3Bwpi7LCNOVL06TV6TTpXup0")
        sys.exit(1)
    
    client_id = sys.argv[1]
    is_valid = verify_client_id(client_id)
    sys.exit(0 if is_valid else 1)

