from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prediction_helper import predict

#Defining input parameters for the model
class CreditRiskInput(BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_tenure_months: int
    avg_dpd_per_delinquency: float
    delinquency_ratio: float
    credit_utilization_ratio: float
    num_open_accounts: int
    residence_type: str
    loan_purpose: str
    loan_type: str

# Defining output parameters for the model
class CreditRiskOutput(BaseModel):
    probability: float
    credit_score: int
    rating: str
app = FastAPI()

@app.get("/ping")
def ping():
    return "Hello, I am alive!"

#app.get cannot get data for complex json objects(more number of input parameters),i.e, we need to use app.post.
@app.post("/predict_credit_risk", response_model=CreditRiskOutput)
def predict_credit_risk(input_data: CreditRiskInput):
    print("Request received")
    try:
        probability, credit_score, rating = predict(input_data.age, input_data.income, input_data.loan_amount,
                                                    input_data.loan_tenure_months, input_data.avg_dpd_per_delinquency,
                                                    input_data.delinquency_ratio, input_data.credit_utilization_ratio,
                                                    input_data.num_open_accounts, input_data.residence_type,
                                                    input_data.loan_purpose, input_data.loan_type)
        return CreditRiskOutput(probability=probability, credit_score=credit_score, rating=rating)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))