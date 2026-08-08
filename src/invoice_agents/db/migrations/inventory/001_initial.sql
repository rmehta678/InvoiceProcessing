CREATE TABLE inventory (
    sku TEXT PRIMARY KEY,
    item_name TEXT UNIQUE NOT NULL,
    available_stock INTEGER NOT NULL CHECK (available_stock >= 0)
);

CREATE TABLE item_aliases (
    alias_normalized TEXT UNIQUE NOT NULL,
    sku TEXT NOT NULL REFERENCES inventory(sku),
    source TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL
);

CREATE INDEX idx_item_aliases_sku ON item_aliases(sku);

