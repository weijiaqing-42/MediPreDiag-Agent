CREATE TABLE IF NOT EXISTS users (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    phone       VARCHAR(20) UNIQUE,
    nickname    VARCHAR(64),
    gender      TINYINT,
    birth_year  INT,
    allergy_info TEXT,
    chronic_conditions TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id          VARCHAR(36) PRIMARY KEY,
    user_id     BIGINT,
    start_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time    DATETIME,
    intent_path JSON,
    is_emergency BOOLEAN DEFAULT FALSE,
    status      ENUM('active','completed','interrupted','timeout') DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id      VARCHAR(36),
    user_id         BIGINT,
    extracted_symptoms JSON,
    possible_diseases  JSON,
    severity_level  ENUM('mild','moderate','severe','unknown'),
    medical_advice  TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS drug_queries (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id     BIGINT,
    drug_name   VARCHAR(128),
    query_content TEXT,
    response    TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);