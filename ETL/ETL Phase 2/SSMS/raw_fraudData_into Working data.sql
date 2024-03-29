USE [Statistical Anomaly Detection]
GO
/****** Object:  StoredProcedure [dbo].[Raw_FraudData_to_WRK]    Script Date: 10/1/2023 11:49:18 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

ALTER PROCEDURE [dbo].[Raw_FraudData_to_WRK]
-- =============================================
-- Author:		Pratham Chawla
-- Create date: October 1, 2023
-- Description:	Raw Data to Working Data
-- =============================================
AS
BEGIN
	-- =============================================
	-- DROP TABLE 
	-- =============================================

	IF OBJECT_ID('WRK_FraudData') IS NOT NULL
	DROP TABLE [WRK_FraudData]

	-- =============================================
	--CREATE TABLE 
	-- =============================================
	CREATE TABLE [WRK_FraudData](
	   [type]                 VARCHAR(100)
      ,[amount]               FLOAT
      ,[oldbalanceOrg]        VARCHAR(100)
      ,[newbalanceOrig]       VARCHAR(100)
      ,[oldbalanceDest]       VARCHAR(100) 
      ,[newbalanceDest]       VARCHAR(100)
      ,[isFraud]              INT
	)

		-- =============================================
	-- TRUNCATE TABLE 
	-- =============================================

	TRUNCATE TABLE [WRK_FraudData]

	-- =============================================
	-- INSERT INTO
	-- =============================================
	INSERT INTO [WRK_FraudData](
	 [type]                 
      ,[amount]               
      ,[oldbalanceOrg]       
      ,[newbalanceOrig]        
      ,[oldbalanceDest]       
      ,[newbalanceDest]       
      ,[isFraud] 
	  )
	  SELECT 
	   [type]                 
      ,[amount]               
      ,[oldbalanceOrg]       
      ,[newbalanceOrig]        
      ,[oldbalanceDest]       
      ,[newbalanceDest]       
      ,[isFraud] 

	  FROM [Fraud Data]

-- (1048567 rows affected)


END
