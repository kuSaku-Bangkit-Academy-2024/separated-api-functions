const functions = require('@google-cloud/functions-framework');
const { Firestore } = require('@google-cloud/firestore');

functions.http('updateMonthlyIncome', async (_req, res) => {
  const db = new Firestore({
    projectId: process.env.PROJECT_ID,
    databaseId: process.env.DATABASE
  });
  const now = new Date();
  now.setHours(now.getHours() + 7);
  const data = [];

  const usersSnapshot = await db.collection('users').get();

  for (const userDoc of usersSnapshot.docs) {
      const walletSnapshot = await userDoc.ref.collection('wallets').get();
      const userData = userDoc.data();
      const currentIncome = userData.income;

      const batch = db.batch();
      walletSnapshot.forEach(walletDoc => {
          const walletRef = walletDoc.ref;
          
          batch.update(walletRef, { income: currentIncome, balance: currentIncome, totalExpense: 0 });
      });

      await batch.commit();
  }
  res.send("success");
});
