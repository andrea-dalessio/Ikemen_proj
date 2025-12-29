bash ./compile.sh || {
    echo "Compile failed"
    exit 1
}

bash ./start.sh
