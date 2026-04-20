from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
                     path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
                     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
                     path("SignupAction", views.SignupAction, name="SignupAction"),
                     path("Signup.html", views.Signup, name="Signup"),
                     path("GenerateNarative.html", views.GenerateNarative, name="GenerateNarative"),
	             path("GenerateNarativeAction", views.GenerateNarativeAction, name="GenerateNarativeAction"),	    
		     path("GenerateQuestion.html", views.GenerateQuestion, name="GenerateQuestion"),
	             path("GenerateQuestionAction", views.GenerateQuestionAction, name="GenerateQuestionAction"),	
		     path("LoadModel", views.LoadModel, name="LoadModel"),
		     path("CourseRecommend.html", views.CourseRecommend, name="CourseRecommend"),
	             path("CourseRecommendAction", views.CourseRecommendAction, name="CourseRecommendAction"),
]
