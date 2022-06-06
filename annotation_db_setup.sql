DROP DATABASE IF EXISTS lung;
CREATE DATABASE lung;
USE lung;
DROP TABLE IF EXISTS bronchus_annotation;
CREATE TABLE bronchus_annotation(
    id VARCHAR(32) PRIMARY KEY,
    file_name VARCHAR(50) NOT NULL, 
    xmin FLOAT NOT NULL, 
    ymin FLOAT NOT NULL, 
    xmax FLOAT NOT NULL,
    ymax FLOAT NOT NULL,
    a FLOAT,
    b FLOAT,
    c FLOAT
);
DROP TABLE IF EXISTS vessel_annotation;
CREATE TABLE vessel_annotation(
    id VARCHAR(32) PRIMARY KEY,
    file_name VARCHAR(50) NOT NULL, 
    xmin FLOAT NOT NULL, 
    ymin FLOAT NOT NULL, 
    xmax FLOAT NOT NULL,
    ymax FLOAT NOT NULL,
    d FLOAT
);
DROP TABLE IF EXISTS overall_annotation;
CREATE TABLE overall_annotation(
    id VARCHAR(32) PRIMARY KEY,
    file_name VARCHAR(50) NOT NULL, 
    e FLOAT
)

