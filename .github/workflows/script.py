import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import *
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import nltk
nltk.download('vader_lexicon')

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# STEP 1

bussdf = spark.read.json("s3://yelp-input-data-g3/yelp_academic_dataset_business.json")

attributes_to_check = [
     "AcceptsInsurance", "AgesAllowed", "Alcohol", "Ambience", "BYOB",
     "ByAppointmentOnly", "BusinessAcceptsCreditCards", "HairSpecializesIn",
     "BYOBCorkage", "Corkage", "BestNights", "GoodForDancing", "Music",
     "Smoking", "BusinessAcceptsBitcoin", "GoodForMeal", "NoiseLevel",
     "DriveThru", "RestaurantsGoodForGroups", "RestaurantsTableService",
     "RestaurantsAttire", "GoodForKids", "DogsAllowed", "RestaurantsReservations",
     "HasTV", "OutdoorSeating", "HappyHour", "WheelchairAccessible", "BusinessParking",
     "WiFi", "Caters", "RestaurantsDelivery", "RestaurantsTakeOut", "CoatCheck",
     "RestaurantsPriceRange2", "BikeParking"
 ]
concat_attributes = udf(lambda attrs: ", ".join([attr for attr in attributes_to_check if attrs[attr]] if attrs else []), StringType())
yelp_df = bussdf.withColumn("present_attributes", concat_attributes(col("attributes")))
concat_expr = concat_ws(", ", *[expr(f"'{attr}: ' || attributes.{attr}") for attr in attributes_to_check])
yelp_df = yelp_df.withColumn("Values_attributes", concat_expr)


#STEP - 2


def categorize(categories):
    if categories is None:
        return "Others"
        
    cat_map = {
         ("Pizza", "Italian", "Sushi", "Mexican", "Fast Food", "Seafood", "Indian", "Chinese", "Thai", "Burgers", "Steakhouses", "Restaurants", "Chicken Wings"): "Restaurants",
         ("Schools", "Colleges", "Universities", "Tutoring", "Libraries", "Educational Services", "Art Schools", "Language Schools", "Arts & Entertainment"): "Education",
         ("Hospitals", "Clinics", "Dentists", "Doctors", "Health Practitioners", "Chiropractors", "Optometrists", "Pharmacies", "Medical Centers"): "Health & Medical",
         ("Clothing", "Electronics", "Furniture", "Jewelry", "Department Stores", "Boutiques", "Malls", "Shoes", "Accessories", "Home Decor"): "Shopping",
         ("Salons", "Hair Stylists", "Nail Salons", "Day Spas", "Skin Care", "Beauty Services", "Waxing", "Massage", "Barber Shops"): "Beauty & Spas",
         ("Bars", " Pubs", "Clubs", "Lounges", "Cocktail Bars", "Karaoke", "Dance Clubs", "Wine Bars", "Breweries", "Nightlife"): "Nightlife",
         ("Auto Repair", "Car Dealerships", "Auto Parts", "Tires", "Oil Change", "Car Wash", "Auto Detailing", "Car Rental", "Automotive"): "Automotive",
         ("Gyms", "Yoga", "Pilates", "Personal Trainers", "Fitness Centers", "Martial Arts", "Dance Studios", "CrossFit", "Cycling Classes"): "Fitness & Instruction",
         ("Plumbers", "Electricians", "Contractors", "Landscaping", "Cleaning Services", "Pest Control", "Movers", "Handyman", "Home Services"): "Home Services",
         ("Pet Stores", "Veterinarians", "Pet Grooming", "Pet Sitting", "Pet Training", "Pet Adoption", "Pet Supplies"): "Pets",
         ("Public Services & Government", "Landmarks & Historical Buildings"): "Government Services"
     }
     
    category_list = [category.strip() for category in categories.split(",")]
    for key, value in cat_map.items():
        if any(category in key for category in category_list):
            return value
    return "Others"
     
categorize_udf = udf(categorize, StringType())
yelp_df = yelp_df.withColumn("Category", categorize_udf("categories"))


# STEP 3 Hours 


days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for day in days_of_week:
     yelp_df = yelp_df.withColumn(f"{day}_hours", col("hours").getField(day))
     print(f"Added column: {day}_hours")
     


# STEP 4

checkin_df = spark.read.json("s3://yelp-input-data-g3/yelp_academic_dataset_checkin.json")
checkin_df = checkin_df.withColumn("date_time", split(col("date"), ","))
checkin_df = checkin_df.withColumn("date_time_exploded", explode(col("date_time")))
checkin_df = checkin_df.withColumn("timestamp", col("date_time_exploded").cast("timestamp"))
checkin_df = checkin_df.withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))
checkin_df = checkin_df.withColumn("time", date_format(col("timestamp"), "HH:mm:ss"))
checkin_df = checkin_df.select("business_id", "date", "time")

# STEP - 5
tips_df = spark.read.json("s3://yelp-input-data-g3/yelp_academic_dataset_tip.json")

preprocessed_tips = tips_df.withColumn("preprocessed_text",
    split(lower(regexp_replace(col("text"), "[^a-zA-Z\\s]", "")), " ")
)
remover = StopWordsRemover(inputCol="preprocessed_text", outputCol="words")
words_removed = remover.transform(preprocessed_tips)
exploded_words = words_removed.select("business_id", "user_id", "text", explode("words").alias("word"))
word_counts = exploded_words.groupBy("business_id", "user_id", "text", "word").count()
window_spec = Window.partitionBy("business_id", "user_id").orderBy(col("count").desc())
ranked_word_counts = word_counts.withColumn("rank", row_number().over(window_spec))
top_words_per_business_user = ranked_word_counts.filter(col("rank") <= 10)
grouped_words = top_words_per_business_user.groupBy("business_id", "user_id", "text").agg(
    concat_ws(", ", collect_list("word")).alias("top_words")
)
grouped_words.show(truncate=False)

sia = SentimentIntensityAnalyzer()
def vader_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
            return 'positive'
    elif compound_score <= -0.05:
            return 'negative'
    else:
            return 'neutral'
vader_sentiment_udf = udf(vader_sentiment, StringType())
tip_sent = grouped_words.withColumn("sentiment", vader_sentiment_udf(grouped_words["text"]))

# STEP - 6

rev = spark.read.json('s3://yelp-input-data-g3/yelp_academic_dataset_review.json')
rev_sentiment = rev.withColumn("sentiment", vader_sentiment_udf(rev["text"]))

# STEP - 7

userdf = spark.read.json("s3://yelp-input-data-g3/yelp_academic_dataset_user.json")
df = userdf.withColumn("yelping_since_timestamp", to_timestamp(col("yelping_since"), "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("yelping_since_date", date_format(col("yelping_since_timestamp"), "yyyy-MM-dd"))

split_expr = split(df['elite'], ',')
df = df.withColumn('elite_count', size(split_expr))

split_friends_expr = split(df['friends'], ',')
df = df.withColumn('num_friends', size(split_friends_expr))
df = df.withColumn('count_friends', col('num_friends'))

columns_to_trim = [col_name for col_name, col_type in df.dtypes if col_type == 'string']
for col_name in columns_to_trim:
    df = df.withColumn(col_name, trim(col(col_name)))


yelp_df = yelp_df.repartition(1)
tip_sent = tip_sent.repartition(1)
checkin_df = checkin_df.repartition(1)
rev_sentiment = rev_sentiment.repartition(1)
df = df.repartition(1)


yelp_df.write.json("s3://yelp-output-data-g3/business_final/", mode = "overwrite")
tip_sent.write.json("s3://yelp-output-data-g3/tips_final/", mode = "overwrite")
checkin_df.write.json("s3://yelp-output-data-g3/checkin_final/", mode = "overwrite")
rev_sentiment.write.json("s3://yelp-output-data-g3/review_final/", mode = 'overwrite')
df.write.json("s3://yelp-output-data-g3/user_final/", mode = 'overwrite')

job.commit()