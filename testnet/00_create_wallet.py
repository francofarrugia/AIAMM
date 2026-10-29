"""Step 0 – create a NEW, empty test wallet for the Sepolia demonstration.
Never use this wallet for real funds. The private key is stored in wallet.json – keep it private."""
import json, secrets
from pathlib import Path
from eth_account import Account

p = Path(__file__).resolve().parent / 'wallet.json'
if p.exists():
    w = json.loads(p.read_text())
    print('wallet.json already exists. Address:', w['address'])
else:
    acct = Account.create(secrets.token_hex(32))
    p.write_text(json.dumps({'address': acct.address, 'private_key': acct.key.hex()}, indent=2))
    print('New test wallet created and saved to wallet.json')
    print('Address:', acct.address)
print('\nNext: request free Sepolia test ETH for this address from a faucet (see README), then run  python 01_deploy.py')
