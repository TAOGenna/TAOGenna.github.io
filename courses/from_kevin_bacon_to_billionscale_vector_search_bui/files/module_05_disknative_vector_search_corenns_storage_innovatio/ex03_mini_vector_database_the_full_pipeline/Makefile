.PHONY: test test-solution clean help

help:
	@echo "Mini Vector Database — Exercise 3"
	@echo ""
	@echo "Targets:"
	@echo "  make test          Run tests against your implementation (vectordb/index.py)"
	@echo "  make test-solution Run tests against the provided solution (_solutions/index.py)"
	@echo "  make clean         Remove __pycache__ directories"

test:
	@echo "Running tests against vectordb/index.py..."
	python -m pytest tests/ -v

test-solution:
	@echo "Running tests against _solutions/index.py..."
	@cp vectordb/index.py vectordb/index.py.backup 2>/dev/null || true
	@cp _solutions/index.py vectordb/index.py
	python -m pytest tests/ -v
	@cp vectordb/index.py.backup vectordb/index.py 2>/dev/null || true
	@rm -f vectordb/index.py.backup

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
