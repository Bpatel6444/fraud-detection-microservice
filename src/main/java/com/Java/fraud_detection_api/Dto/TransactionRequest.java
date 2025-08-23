package com.Java.fraud_detection_api.Dto;

public class TransactionRequest {
    private Double amount;
    private Integer hourOfDay;

    // Default constructor
    public TransactionRequest() {
    }

    // Parameterized constructor
    public TransactionRequest(Double amount, Integer hourOfDay) {
        this.amount = amount;
        this.hourOfDay = hourOfDay;
    }

    // Getters and setters
    public Double getAmount() {
        return amount;
    }

    public void setAmount(Double amount) {
        this.amount = amount;
    }

    public Integer getHourOfDay() {
        return hourOfDay;
    }

    public void setHourOfDay(Integer hourOfDay) {
        this.hourOfDay = hourOfDay;
    }
}