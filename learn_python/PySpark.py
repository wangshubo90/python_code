sparkSession = SQLContext.getOrCreate(sc).sparkSession

rdd = sc.parallelize([101] + range(100))
rdd.map(lambda x: x ** x).take(10).reduce(lambda a,b: a+b)

sum = rdd.sum()
n = rdd.count()
mean = sum / n

sortedAndIndexed = rdd.sortBy(lambda x:x).collect().zipWithIndex().map(lambda (value, key) : (key, value)).collect()

def find_median(rdd):
    n = rdd.count()
    if n % 2 == 1:
        index = (n - 1) / 2
        median = rdd.lookup(index)
    else:
        index1 = (n/2)-1
        index2 = (n/2)
        median = (rdd.lookup(index1) + rdd.lookup(index2))
    return median 

from math import sqrt
std = sqrt( 
    rdd.map(
        lambda x: (x-mean) ** 2
        ).sum() 
)

