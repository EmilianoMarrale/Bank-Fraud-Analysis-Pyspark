import random
from functools import reduce
from pyspark.sql import DataFrame, SparkSession, Row
import pyspark.sql.functions as F
from pyspark.sql.functions import array, create_map, struct, rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql.window import Window
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, LongType
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer,StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector, DenseVector
from pyspark.ml import Pipeline

def pre_smote_df_process(df,num_cols,cat_cols,target_col,index_suffix="_index"):
    '''
    string indexer and vector assembler
    inputs:
    * df: spark df, original
    * num_cols: numerical cols to be assembled
    * cat_cols: categorical cols to be stringindexed
    * target_col: prediction target
    * index_suffix: will be the suffix after string indexing
    output:
    * vectorized: spark df, after stringindex and vector assemble, ready for smote
    '''
    if(df.select(target_col).distinct().count() != 2):
        raise ValueError("Target col must have exactly 2 classes")

    if target_col in num_cols:
        num_cols.remove(target_col)

    # only assembled numeric columns into features
    assembler = VectorAssembler(inputCols = num_cols, outputCol = 'features_for_model')
    # index the string cols, except possibly for the label col
    assemble_stages = [StringIndexer(inputCol=column, outputCol=column+index_suffix) for column in list(set(cat_cols)-set([target_col]))]
    # add the stage of numerical vector assembler
    assemble_stages.append(assembler)
    pipeline = Pipeline(stages=assemble_stages)
    pos_vectorized = pipeline.fit(df).transform(df)

    # drop original num cols and cat cols
    drop_cols = num_cols+cat_cols

    keep_cols = [a for a in pos_vectorized.columns if a not in drop_cols]

    vectorized = pos_vectorized.select(*keep_cols).withColumn('label',pos_vectorized[target_col]).drop(target_col)

    return vectorized

def subtract_vector_fn(arr):
    a = arr[0]
    b = arr[1]

    if isinstance(a, SparseVector):
        a = a.toArray()

    if isinstance(b, SparseVector):
        b = b.toArray()

    return DenseVector(random.uniform(0, 1)*(a-b))

def add_vector_fn(arr):
    a = arr[0]
    b = arr[1]

    if isinstance(a, SparseVector):
        a = a.toArray()

    if isinstance(b, SparseVector):
        b = b.toArray()

    return DenseVector(a+b)

def smote(vectorized_sdf,smote_config):
    '''
    contains logic to perform smote oversampling, given a spark df with 2 classes
    inputs:
    * vectorized_sdf: cat cols are already stringindexed, num cols are assembled into 'features' vector
      df target col should be 'label'
    * smote_config: config obj containing smote parameters
    output:
    * oversampled_df: spark df after smote oversampling
    '''
    dataInput_min = vectorized_sdf[vectorized_sdf['label'] == smote_config.positive_label]
    dataInput_maj = vectorized_sdf[vectorized_sdf['label'] == smote_config.negative_label]

    # LSH, bucketed random projection
    brp = BucketedRandomProjectionLSH(inputCol="features_for_model", outputCol="hashes",seed=int(smote_config.seed), \
                                      bucketLength=float(smote_config.bucketLength))
    # smote only applies on existing minority instances
    model = brp.fit(dataInput_min)
    model.transform(dataInput_min)

    # here distance is calculated from brp's param inputCol
    self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float('inf'), distCol="EuclideanDistance")

    # remove self-comparison (distance 0)
    self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)

    over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")

    self_similarity_df = self_join_w_distance.withColumn("r_num", F.row_number().over(over_original_rows))

    self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= int(smote_config.k))

    over_original_rows_no_order = Window.partitionBy('datasetA')

    # list to store batches of synthetic data
    res = []

    # two udf for vector add and subtract, subtraction include a random factor [0,1]
    subtract_vector_udf = F.udf(subtract_vector_fn, VectorUDT())
    add_vector_udf = F.udf(add_vector_fn, VectorUDT())

    # retain original columns
    original_cols = dataInput_min.columns

    for i in range(int(smote_config.multiplier)):
        print("generating batch %s of synthetic instances"%i)
        # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
        df_random_sel = self_similarity_df_selected \
            .withColumn("rand", F.rand()) \
            .withColumn('max_rand', F.max('rand').over(over_original_rows_no_order)) \
            .where(F.col('rand') == F.col('max_rand')).drop(*['max_rand','rand','r_num'])
        # create synthetic feature numerical part
        df_vec_diff = df_random_sel \
            .select('*', subtract_vector_udf(F.array('datasetA.features_for_model', 'datasetB.features_for_model')).alias('vec_diff'))
        df_vec_modified = df_vec_diff \
            .select('*', add_vector_udf(F.array('datasetB.features_for_model', 'vec_diff')).alias('features_for_model'))

        # for categorical cols, either pick original or the neighbour's cat values
        for c in original_cols:
            # randomly select neighbour or original data
            col_sub = random.choice(['datasetA','datasetB'])
            val = "{0}.{1}".format(col_sub,c)
            if c != 'features_for_model':
                # do not unpack original numerical features
                df_vec_modified = df_vec_modified.withColumn(c,F.col(val))

        # this df_vec_modified is the synthetic minority instances,
        df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])

        res.append(df_vec_modified)

    dfunion = reduce(DataFrame.union, res)
    dfunion = dfunion.union(dataInput_min.select(dfunion.columns)) \
        .sort(F.rand(seed=smote_config.seed)) \
        .withColumn('row_number', row_number().over(Window.orderBy(lit('A'))))

    dataInput_maj = dataInput_maj.withColumn('row_number', row_number().over(Window.orderBy(lit('A'))))

    # union synthetic instances with original full (both minority and majority) df
    oversampled_df = dfunion.union(dataInput_maj.select(dfunion.columns))

    return oversampled_df.sort('row_number').drop(*['row_number'])

class SmoteConfig:
    def __init__(self, seed, bucketLength, k, multiplier, positive_label, negative_label):
        self.seed = seed
        self.bucketLength = bucketLength
        self.k = k
        self.multiplier = multiplier
        self.positive_label = positive_label
        self.negative_label = negative_label


