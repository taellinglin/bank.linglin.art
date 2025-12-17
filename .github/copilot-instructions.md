<!-- Copilot / AI agent instructions for contributors working on bank.linglin.art -->

# Quick Orientation

- **Big picture**: This repository implements a lightweight blockchain-based banknote system. Core responsibilities are mining/validating blocks, managing a mempool, creating genesis "GTX_Genesis" bills, and serving a web UI/API.
- **Primary components**:
  - `blockchain_daemon.py`: central daemon that manages local `blockchain.json`, `mempool.json`, syncing with network endpoints, mining-validation logic, and interactions with `lunalib` managers.
  - `app.py`, `front.py`, `wallet_server.py`, `gui_wallet.py`: UI / API entrypoints and web frontends (templates in `templates/`, static assets in `static/`).
  - `lunalib` usage: `BlockchainManager`, `MempoolManager`, `GTXGenesis`, `DigitalBill`, `BillRegistry` are imported and used for network operations and bill handling.
  - Persistence: `blockchain.json`, `mempool.json`, `wallet.json`; DB code lives in `db/` and `migrations/`.

# Key patterns & conventions (be specific)

- Transaction types: `GTX_Genesis`, `genesis`, `transfer`, `reward`. Many validators branch on these exact strings — search `"type": "GTX_Genesis"` when adding features.
- Merkle root: calculated by `_calculate_merkle_root()` in `blockchain_daemon.py`. It uses existing `tx['hash']` or sha256(JSON(tx)) when missing, then pairwise sha256 of concatenated hex strings.
- Block hashing: `calculate_block_hash(...)` uses `json.dumps(..., sort_keys=True, separators=(',',':'))` of a dict with keys `index, previous_hash, timestamp, transactions, nonce` — maintain this structure if producing or validating hashes externally.
- Mining proof: `validate_mining_proof(...)` constructs `block_string = previous_hash + timestamp + merkleroot + miner + nonce` and computes sha256; difficulty = count of leading hex `'0'` characters (1..9). This is non-standard (hex leading zeros, not bit-target); tests and reward logic rely on it.
- Rewards: `validate_reward_transactions(...)` expects `amount == BASE_REWARD * difficulty` (BASE_REWARD currently 1). Changing reward semantics requires updating this function and related validation.
- Validation surface: when adding a new transaction type or changing validation, update these functions in `blockchain_daemon.py`: `validate_transaction_structure`, `validate_regular_transactions`, `validate_transaction_for_block`, and `get_mempool_status` / `get_blockchain_status` if you want it counted.

# Integration points & external dependencies

- Network endpoints: default `endpoint_url` is `https://bank.linglin.art`. `BlockchainManager` and `MempoolManager` are used to fetch blocks, mempool, submit mined blocks, and broadcast transactions.
- `lunalib` is an external/monorepo dependency—ensure it is importable in your environment. The daemon calls `self.mempool_mgr.test_connection()` frequently; network behavior is gated by that.
- Persistent files to watch during development: `blockchain.json`, `mempool.json`, files under `blockchain_daemon/` (contains example local mining files), and `db/` migrations.

# Developer workflows & commands

- Run the web app / API (development):
  - In PowerShell: `python app.py` (or run the specific frontend server file such as `python wallet_server.py`).
- Run the background daemon for local mining/sync: `python blockchain_daemon.py`.
- Build helpers / platform scripts:
  - Windows batch helpers exist: `build.bat`, `start_treasury.bat`, `banknote_generator.bat` — inspect them for platform-specific tasks.
- There are no automated tests in the repo; use targeted runs and logging. Logging is through `logging.getLogger(__name__)` in files like `blockchain_daemon.py` — increase verbosity by setting `logging.basicConfig(level=logging.DEBUG)` during local debugging.

# What to update when changing core behavior

- If you change block hashing, update `calculate_block_hash` and all callers (`validate_block_for_submission`, `validate_block`, `submit_block_with_validation`).
- If you change mining proof semantics (e.g., count leading *bits* instead of hex characters), update `validate_mining_proof` and `validate_reward_transactions` (reward calculation relies on difficulty derived there).
- If you add a transaction type, ensure it's handled consistently in:
  - `validate_transaction_structure` (serialization/required fields)
  - `validate_regular_transactions` and/or `validate_transaction_for_block` (block-inclusion checks)
  - `get_mempool_status` / `get_blockchain_status` (for statistics/UI)

# Files to reference for examples

- `blockchain_daemon.py`: authoritative behaviors for syncing, validation, mining proof, reward calculation, mempool management.
- `app.py` / `front.py`: how the web-facing routes expect data formats and responses.
- `db/` and `migrations/`: DB schema and seed usage.
- `templates/` and `static/`: UI expectations for JSON shapes (useful for API changes).

# Quick guidelines for AI agents

- Be conservative: prefer modifying helpers over changing validation rules inline. Many validation functions are duplicated/overlapping in `blockchain_daemon.py` — prefer centralizing changes.
- Use exact string values and JSON structures as found in the code (transaction `type` strings, `hash` field presence, `serial_number` for `GTX_Genesis`).
- When adding new APIs or fields, update both server-side validation and UI templates under `templates/` to keep the front-end functioning.

If anything here is unclear or you want me to expand examples (e.g., show a PR patch that adds a new transaction type and updates all validators), say which area and I will iterate.
