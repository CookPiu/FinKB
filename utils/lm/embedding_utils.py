"""BGE-M3 稠密 + 稀疏编码。CPU 上加载模型要数十秒，模块级单例，服务启动时调用 warmup() 预热。"""
import time

from common.config.embedding_config import embedding_config
from common.logging.logger import logger

# 模型单例，避免重复加载
_bge_m3_model = None


def get_bge_m3_model():
    """获取 BGE-M3 模型单例（稠密 + 稀疏向量由同一次前向得到）"""
    global _bge_m3_model
    if _bge_m3_model is None:
        # 放在函数里导入：torch 加载较慢，不用向量的命令（如 status）不必付出这个代价
        from FlagEmbedding import BGEM3FlagModel

        start_ts = time.time()
        _bge_m3_model = BGEM3FlagModel(
            embedding_config.bge_m3_path,
            normalize_embeddings=True,
            use_fp16=embedding_config.bge_fp16,
            devices=embedding_config.bge_device,
        )
        logger.info(f"BGE-M3 加载完成，用时 {time.time() - start_ts:.1f}s（device={embedding_config.bge_device}）")
    return _bge_m3_model


def generate_embeddings(texts: list, max_length=None, batch_size=None) -> list:
    """
    编码一批文本
    :param texts: 文本列表
    :param max_length: 最大 token 数，默认取配置 BGE_MAX_LENGTH
    :param batch_size: 每批条数，默认取配置 BGE_BATCH_SIZE
    :return: 与 texts 一一对应的 [{"dense": [1024 个 float], "sparse": {维度(int): 权重(float)}}]
    """
    if not texts:
        return []
    if max_length is None:
        max_length = embedding_config.bge_max_length
    if batch_size is None:
        batch_size = embedding_config.bge_batch_size
    output = get_bge_m3_model().encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    results = []
    for dense, lexical_weights in zip(output["dense_vecs"], output["lexical_weights"]):
        # 稀疏向量的键是字符串形式的 token id，Milvus 要求整数键；权重为 0 的维度去掉
        sparse = {}
        for token_id, weight in lexical_weights.items():
            if float(weight) > 0:
                sparse[int(token_id)] = float(weight)
        results.append({"dense": [float(x) for x in dense], "sparse": sparse})
    return results


def generate_query_embedding(text: str) -> dict:
    """编码一个问题；问题较短，max_length 取 512"""
    return generate_embeddings([text], max_length=512, batch_size=1)[0]


def warmup():
    """预热：触发模型加载并完成一次编码，避免首个用户请求等待"""
    generate_query_embedding("预热")
