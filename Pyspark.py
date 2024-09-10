# Databricks notebook source
df1=spark.read.parquet('dbfs:/FileStore/tables/part_00000_ab7d4ae4_a0cb_4af9_b216_15fa8184cdf9_c000_snappy-3.parquet')
df2=spark.read.parquet('dbfs:/FileStore/tables/part_00001_ab7d4ae4_a0cb_4af9_b216_15fa8184cdf9_c000_snappy-3.parquet')
df3=spark.read.parquet('dbfs:/FileStore/tables/part_00000_76e3d321_e17b_4d57_bdc7_05a21b133269_c000_snappy-3.parquet')
df4=spark.read.parquet('dbfs:/FileStore/tables/part_00001_76e3d321_e17b_4d57_bdc7_05a21b133269_c000_snappy-3.parquet')
df5=spark.read.parquet('dbfs:/FileStore/tables/part_00002_e991df75_1ab7_447e_8b8c_761c4130320f_c000_snappy-3.parquet')
df6=spark.read.parquet('dbfs:/FileStore/tables/part_00000_2e6a9f86_0460_4d04_935b_9a5d6ea7ef04_c000_snappy-3.parquet')
df7=spark.read.parquet('dbfs:/FileStore/tables/part_00001_e991df75_1ab7_447e_8b8c_761c4130320f_c000_snappy-3.parquet')
df8=spark.read.parquet('dbfs:/FileStore/tables/part_00000_e991df75_1ab7_447e_8b8c_761c4130320f_c000_snappy-3.parquet')
df9=spark.read.parquet('dbfs:/FileStore/tables/part_00001_2e6a9f86_0460_4d04_935b_9a5d6ea7ef04_c000_snappy-2.parquet')
df10=spark.read.parquet('dbfs:/FileStore/tables/part_00001_11635212_ab6b_4984_a903_26e4d2a2a1f9_c000_snappy-2.parquet')
df11=spark.read.parquet('dbfs:/FileStore/tables/part_00000_11635212_ab6b_4984_a903_26e4d2a2a1f9_c000_snappy-2.parquet')
df12=spark.read.parquet('dbfs:/FileStore/tables/part_00000_eb119b42_62fb_4dfe_8d3e_e1726fb27f1d_c000_snappy-3.parquet')
df13=spark.read.parquet('dbfs:/FileStore/tables/part_00001_6796266b_c6b3_42fb_87ec_198f5b608d01_c000_snappy-2.parquet')
df14=spark.read.parquet('dbfs:/FileStore/tables/part_00000_6796266b_c6b3_42fb_87ec_198f5b608d01_c000_snappy-2.parquet')


# Funciones a usar
from pyspark.sql.functions import lit, to_date,row_number, desc,col, to_timestamp, date_format
from pyspark.sql.window import Window
date_format = 'yyyyMMdd'

df1_con_fecha = df1.withColumn("Fecha", to_date(lit("20240701"), date_format))
df2_con_fecha = df2.withColumn("Fecha", to_date(lit("20240701"), date_format))
df3_con_fecha = df3.withColumn("Fecha", to_date(lit("20240702"), date_format))
df4_con_fecha = df4.withColumn("Fecha", to_date(lit("20240702"), date_format))
df5_con_fecha = df5.withColumn("Fecha", to_date(lit("20240703"), date_format))
df6_con_fecha = df6.withColumn("Fecha", to_date(lit("20240703"), date_format))
df7_con_fecha = df7.withColumn("Fecha", to_date(lit("20240703"), date_format))
df8_con_fecha = df8.withColumn("Fecha", to_date(lit("20240703"), date_format))
df9_con_fecha = df9.withColumn("Fecha", to_date(lit("20240703"), date_format))
df10_con_fecha = df10.withColumn("Fecha", to_date(lit("20240705"), date_format))
df11_con_fecha = df11.withColumn("Fecha", to_date(lit("20240705"), date_format))
df12_con_fecha = df12.withColumn("Fecha", to_date(lit("20240706"), date_format))
df13_con_fecha = df13.withColumn("Fecha", to_date(lit("20240707"), date_format))
df14_con_fecha = df14.withColumn("Fecha", to_date(lit("20240707"), date_format))


# Creo lista de dataframes y las uno ----------


dfs = [df1_con_fecha,df2_con_fecha,df3_con_fecha,df4_con_fecha,df5_con_fecha,df6_con_fecha,
df7_con_fecha,df8_con_fecha,df9_con_fecha,df10_con_fecha,df11_con_fecha,df12_con_fecha,
df13_con_fecha,df14_con_fecha]
res=dfs[0]
for df in dfs[1:]:
    res=res.unionByName(df,allowMissingColumns=True)

res.fillna(0).fillna("")

df_final=res.withColumn("facturacion", col("price") * col("sales") )


# PUNTO 3A: SE USO ID como vendedor porque no existe la columna SELLERSID  ----------

df_vendedor=df_final.groupBy('Id').sum('facturacion')

# PUNTO 3B: Se uso title como nombre del articulo  ----------

df_alternativo = df_final.filter(col("title") != "")
df_alternativo.groupby('title').sum('sales')

#Rankeo de vendedores
window_spec = Window.orderBy(desc("sum(facturacion)"))

df_ranking = df_vendedor.withColumn("row_num", row_number().over(window_spec))
df_ranking.write.csv('Punto4.csv')


# Particion año,mes e id ----------

df_final2 =df_final.withColumn("Fecha", to_timestamp(col("Fecha"), 'yyyyMMdd')) \
           .withColumn("year", date_format(col("Fecha"), "yyyy"))\
           .withColumn("month",date_format(col("Fecha"),"MM"))
df_final2.write.partitionBy("year", "month", "id") \
    .mode("overwrite") \
    .parquet("/dbfs/FileStore/tables/destination/")
