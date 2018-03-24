Page({

  /**
   * 页面的初始数据
   */
  data: {
    waitmsg:'请骚后2...',
    status: 'waiting'
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {
    wx.request({
      url: 'https://528393028.peitaiyi.xyz ',
      method: 'GET',
      success: (res) => {
        if (+res.statusCode == 200) {
          console.log('服务器响应成功' + res.data)
          this.setData({ waitmsg: '服务器响应成功!', status:'success'})
        }
        else {
          this.setData({ waitmsg: '服务器无响应!', status:'success'})
        }
        }
    });
  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {
 
  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {
    
  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {
    
  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {
  }
})