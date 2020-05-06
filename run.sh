#!/usr/bin/env bash
echo "Tworzenie bazy danych: "
python3 loadDataToCSV.py
echo "Obróbka plików: "
mkdir -p clear
python3 cleaning.py


