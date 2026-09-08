CREATE KEYSPACE IF NOT EXISTS object_detection 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE object_detection.results (
    id UUID PRIMARY KEY,
    timestamp timestamp,
    prediction text,
    confidence double
);

