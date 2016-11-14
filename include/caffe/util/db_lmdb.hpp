#ifdef USE_LMDB
#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <string>
#include <vector>

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBCursor : public Cursor {
 public:
  explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor, const int entries)
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false), nr_entries_(entries), entry_counter_(0), epoch_counter_(0) {
    MDB_CHECK(mdb_dbi_open(mdb_txn_, NULL, 0, &mdb_dbi_));
    readAllKeys();
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
    mdb_cursor_close(mdb_cursor_);
    mdb_txn_abort(mdb_txn_);
  }
  virtual void SeekToFirst() { Seek(MDB_FIRST); }
  virtual void Next() { Seek(MDB_NEXT); }
  virtual string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  virtual string value() {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual bool valid() { return valid_; }

  /* lingni implemented
   * this function grabs a record from lmdb by its associated key value
   * which provides the support to shuffle lmdb
   */
  virtual void SeekOneByKey()
  {
    // increment epoch counter and reset entry counter
    if (entry_counter_ >= nr_entries_)
    {
      std::random_shuffle(all_keys_.begin(), all_keys_.end());
      ++epoch_counter_;
      entry_counter_ = 0;
      LOG(INFO) << "shuffle data, start epoch " << epoch_counter_;
    }

    mdb_key_.mv_data = const_cast<char*> (all_keys_[entry_counter_].data());
    mdb_key_.mv_size = all_keys_[entry_counter_].size();
    ++entry_counter_;

    int mdb_status = mdb_get(mdb_txn_, mdb_dbi_, &mdb_key_, &mdb_value_);
    CHECK_NE(mdb_status, MDB_NOTFOUND) << "fail to find record with key: " << all_keys_[entry_counter_];

    DLOG(INFO) << "LMDB shuffle read: entry_counter = " << entry_counter_ << ", key = " << key();
    valid_ = true;
  }


 private:
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;

      DLOG(INFO)<< "LMDB read image keyvalue = " << key();
    }
  }


  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_;

  // linngi added for shuffeling LMDB
  MDB_dbi mdb_dbi_;                  // recieved from parent LMDB
  int nr_entries_;
  int entry_counter_;                 // keep track of entries to read
  int epoch_counter_;                 // keep track how many epochs have been used
  std::vector<std::string> all_keys_;


  void readAllKeys();
};

class LMDBTransaction : public Transaction {
 public:
  explicit LMDBTransaction(MDB_env* mdb_env)
    : mdb_env_(mdb_env) { }
  virtual void Put(const string& key, const string& value);
  virtual void Commit();

 private:
  MDB_env* mdb_env_;
  vector<string> keys, values;

  void DoubleMapSize();

  DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
};

class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL) { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  virtual LMDBCursor* NewCursor();
  virtual LMDBTransaction* NewTransaction();

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB
