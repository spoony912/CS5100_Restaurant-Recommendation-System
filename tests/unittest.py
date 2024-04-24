# from models.business import Business
from ..models import business
from ..models import user
business = business ("name", "id")
user = user("username", "usersID")
print(business.name)
print(user.name)