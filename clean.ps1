cargo fmt

cargo clippy

Get-ChildItem -Recurse -File ./src, ./scenes | ForEach-Object {
    unix2dos $_.FullName
}
