# $current_path = Split-Path -Parent $MyInvocation.MyCommand.Definition

./venv/Scripts/activate.ps1

$env:FLASK_ENV = "development"
$env:FLASK_APP = "lung"
flask run