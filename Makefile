PYTHON = python3.10
DATA_LOADER = ./dataLoader.py

# 默认目标
.PHONY: all
all: cache

# 清理缓存
.PHONY: clean-cache
clean-cache:
	-rm -rf ./cache/**

# 生成缓存
.PHONY: cache
cache:
	$(PYTHON) $(DATA_LOADER) cache

# 清理输出
.PHONY: clean-output
clean-output:
	-rm -rf ./output/**