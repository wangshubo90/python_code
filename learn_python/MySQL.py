import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  password="yourpassword",
  database="mydatabase"
)

mycursor = mydb.cursor()

#Create a table named "customers":
mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")

#You can check if a table exist by listing all tables in your database with the "SHOW TABLES" statement:
mycursor.execute("SHOW TABLES")

for x in mycursor:
    print(x)

#Create primary key when creating the table:
mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")

#Create primary key on an existing table:
mycursor.execute("ALTER TABLE customers ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY")

#Insert a record in the "customers" table:
sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
mycursor.execute(sql, val)

mydb.commit()

print(mycursor.rowcount, "record inserted.")

#Insert Multiple Rows

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = [
  ('Peter', 'Lowstreet 4'),
  ('Amy', 'Apple st 652'),
  ('Hannah', 'Mountain 21'),
  ('Michael', 'Valley 345'),
  ('Sandy', 'Ocean blvd 2'),
  ('Betty', 'Green Grass 1'),
  ('Richard', 'Sky st 331'),
  ('Susan', 'One way 98'),
  ('Vicky', 'Yellow Garden 2'),
  ('Ben', 'Park Lane 38'),
  ('William', 'Central st 954'),
  ('Chuck', 'Main Road 989'),
  ('Viola', 'Sideway 1633')
]

mycursor.executemany(sql, val)

mydb.commit()

print(mycursor.rowcount, "was inserted.")
print("1 record inserted, ID:", mycursor.lastrowid)

#Select all records from the "customers" table, and display the result:
mycursor.execute("SELECT * FROM customers")

myresult = mycursor.fetchall()

for x in myresult:
    print(x)

# select columns and only one row
mycursor.execute("SELECT name, address FROM customers")
myresult = mycursor.fetchone()

# select with where
sql = "SELECT * FROM customers WHERE address ='Park Lane 38'"
# select with where + wildcard
sql = "SELECT * FROM customers WHERE address LIKE '%way%'"

#Prevent SQL Injection
#Escape query values by using the placholder %s method:

sql = "SELECT * FROM customers WHERE address = %s"
adr = ("Yellow Garden 2", )

mycursor.execute(sql, adr)

#sort 
sql = "SELECT * FROM customers ORDER BY name DESC"

#delete 
sql = "DELETE FROM customers WHERE address = 'Mountain 21'"

#Delete a Table
sql = "DROP TABLE customers"
sql = "DROP TABLE IF EXISTS customers"
mycursor.execute(sql)

#update a table
sql = "UPDATE customers SET address = 'Canyon 123' WHERE address = 'Valley 345'"

mycursor.execute(sql)

mydb.commit()

print(mycursor.rowcount, "record(s) affected")

# fetch limited number with offset
mycursor.execute("SELECT * FROM customers LIMIT 5 OFFSET 2")

myresult = mycursor.fetchall()

# inner join
sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  INNER JOIN products ON users.fav = products.id"

"LEFT JOIN"


SELECT
    b.dim_market
  , SUM(a.m_bookings) AS m_bookings
FROM (
  SELECT
      id_listing
    , 1          AS m_bookings
    , m_a        # not used (for illustration only)
    , m_b        # not used (for illustration only)
    , m_c        # not used (for illustration only)
  FROM
    fct_bookings
  WHERE
    ds BETWEEN '{{ last_sunday }}' AND '{{ this_saturday }}'
) a 
JOIN (
  SELECT
      id_listing
    , dim_market
    , dim_x      # not used (for illustration only)
    , dim_y      # not used (for illustration only)
    , dim_z      # not used (for illustration only)
  FROM
    dim_listings
  WHERE
    ds BETWEEN '{{ latest_ds }}'
) b
ON (a.id_listing = b.id_listing)
GROUP BY
  b.dim_market
;